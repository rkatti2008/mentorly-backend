from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from collections import Counter
import gspread
from google.oauth2.service_account import Credentials
import os, json, re, sqlite3, secrets, hashlib

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mentorlygpt.netlify.app",
        "http://localhost:5500",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Google Sheets Setup
# -------------------------------
creds_dict = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
)
sheet = gspread.authorize(creds).open_by_key(os.environ["SHEET_ID"]).sheet1
client_llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -------------------------------
# Persistent Conversation Memory
# -------------------------------
# The app keeps session context in memory and also writes it to a small JSON file.
# On Render this is still lightweight, but it survives ordinary app reloads while
# the filesystem is preserved. For heavier production usage, replace this with
# Redis, Supabase, Postgres, or another persistent store.
SESSION_STORE_PATH = os.environ.get("SESSION_STORE_PATH", "session_memory.json")

def load_session_memory():
    try:
        if os.path.exists(SESSION_STORE_PATH):
            with open(SESSION_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def save_session_memory():
    try:
        with open(SESSION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(SESSION_MEMORY, f, ensure_ascii=False, indent=2)
    except Exception:
        # Never break the API because session persistence failed.
        pass


SESSION_MEMORY = load_session_memory()


# -------------------------------
# User Auth / Login Database
# -------------------------------
# SQLite is used for the first version because it is simple and has no extra
# package dependency. On Render, set USER_DB_PATH to a persistent disk path
# later if you want the login database to survive redeploys/restarts reliably.
USER_DB_PATH = os.environ.get("USER_DB_PATH", "mentorly_users.db")

def get_db_connection():
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            login_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${derived}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, expected = stored_hash.split("$", 2)
        if algorithm != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            120_000,
        ).hex()
        return secrets.compare_digest(derived, expected)
    except Exception:
        return False

def get_user_by_username_or_email(username_or_email: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM users
        WHERE lower(username) = lower(?) OR lower(email) = lower(?)
        """,
        (username_or_email, username_or_email),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_token(auth_token: Optional[str]):
    if not auth_token:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT users.*
        FROM auth_sessions
        JOIN users ON auth_sessions.user_id = users.id
        WHERE auth_sessions.token = ?
        """,
        (auth_token,),
    )
    row = cur.fetchone()

    if row:
        cur.execute(
            "UPDATE auth_sessions SET last_used_at = ? WHERE token = ?",
            (datetime.utcnow().isoformat() + "Z", auth_token),
        )
        conn.commit()

    conn.close()
    return dict(row) if row else None

def create_auth_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow().isoformat() + "Z"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO auth_sessions (token, user_id, created_at, last_used_at)
        VALUES (?, ?, ?, ?)
        """,
        (token, user_id, now, now),
    )
    conn.commit()
    conn.close()

    return token

def record_chat_history(user_id: Optional[int], session_id: str, question: str, answer: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_history (user_id, session_id, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, session_id, question, answer, datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        conn.close()
    except Exception:
        # Chat should never fail because history logging failed.
        pass

init_user_db()


# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    auth_token: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthRequest(BaseModel):
    auth_token: str

# -------------------------------
# High-level Query Type Detection
# -------------------------------
def is_similar_student_query(q: str) -> bool:
    q = str(q).lower()
    return any(k in q for k in [
        "similar student", "similar students", "students similar to me",
        "similar to my profile", "similar profile", "my profile",
        "matches my profile", "match my profile", "closest profiles"
    ])

def is_dashboard_query(q: str) -> bool:
    q = str(q).lower()
    return any(k in q for k in [
        "dashboard", "overall stats", "overall statistics", "database summary",
        "summary of database", "show me stats", "admissions dashboard"
    ])

def is_university_insights_query(q: str) -> bool:
    q = str(q).lower()
    return any(k in q for k in [
        "university insights", "college insights", "insights for", "insights about",
        "applicants to", "applicants for", "tell me about applicants",
        "tell me about students applying", "profile of students admitted",
        "admitted students at", "admits to"
    ])

# -------------------------------
# Intent Classification
# -------------------------------
def classify_intent(q: str) -> str:
    q = q.lower()
    if is_similar_student_query(q):
        return "similarity"
    if is_dashboard_query(q):
        return "dashboard"
    if is_university_insights_query(q):
        return "university_insights"
    if any(k in q for k in ["how many", "count", "number of"]):
        return "analytics"
    if any(k in q for k in ["advice", "guidance", "counsel", "what should i", "how can i improve", "chance", "chances", "recommend"]):
        return "advisory"
    return "hybrid"

# -------------------------------
# Text Helpers
# -------------------------------
def normalize(text: str) -> str:
    return re.sub(r'[\"\'“”.,()\-]', '', str(text).lower()).strip()

def clean_user_query(q: str) -> str:
    q = q.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        q = q[1:-1]
    return q.strip()

# -------------------------------
# University Canonicalization
# -------------------------------
UNIVERSITY_ALIASES = {
    "cornell": ["cornell university"],
    "ucsd": ["university of california san diego", "uc san diego"],
    "mit": ["massachusetts institute of technology"],
    "uc berkeley": ["university of california berkeley", "uc berkeley"]
}

def normalize_university(val: str) -> str:
    val = normalize(val)
    for canon, aliases in UNIVERSITY_ALIASES.items():
        if val == canon or val in aliases:
            return canon
    return val

# -------------------------------
# School Column Logic
# -------------------------------
SCHOOL_COLUMN_HINTS = ["school", "high", "secondary"]

def is_school_column(col_name: str) -> bool:
    col = normalize(col_name)
    return any(hint in col for hint in SCHOOL_COLUMN_HINTS)

def row_has_school(row: dict, school: str) -> bool:
    target = normalize(school)
    for col, cell in row.items():
        if is_school_column(col):
            cell_norm = normalize(cell)
            if target in cell_norm or cell_norm in target:
                return True
    return False

# -------------------------------
# Admit Column Logic
# -------------------------------
ADMIT_INCLUDE_HINTS = ["final", "admit", "admitted", "decision", "accept", "accepted", "result"]
ADMIT_EXCLUDE_HINTS = ["applied", "application", "preference", "choice", "list"]

def is_admit_column(col_name: str) -> bool:
    col = normalize(col_name)
    if any(bad in col for bad in ADMIT_EXCLUDE_HINTS):
        return False
    return any(good in col for good in ADMIT_INCLUDE_HINTS)

def extract_universities(cell: str):
    return [
        normalize_university(u.strip())
        for u in re.split(r"[;,/|]", str(cell))
        if u.strip()
    ]

def get_university_match_names(university: str):
    """
    Return a flexible set of normalized names for a university.

    This is intentionally generic: it works for Cornell, UCSD, MIT,
    and also for universities that are not listed in UNIVERSITY_ALIASES.
    """
    target = normalize_university(university)
    names = {target}

    for canon, aliases in UNIVERSITY_ALIASES.items():
        canon_norm = normalize_university(canon)
        alias_norms = {normalize_university(a) for a in aliases}
        all_names = {canon_norm, *alias_norms}

        if target in all_names or any(target in name or name in target for name in all_names):
            names.update(all_names)

    return {name for name in names if name}

def university_names_match(query_university: str, admitted_university: str) -> bool:
    """
    Flexible admit matching for all universities, not only hardcoded aliases.

    Examples it handles:
    - Cornell vs Cornell University
    - UCSD vs University of California San Diego
    - University of California Berkeley vs UC Berkeley
    - Any other university where one name is contained in the other
    """
    admitted = normalize_university(admitted_university)
    if not admitted:
        return False

    for target in get_university_match_names(query_university):
        if admitted == target or target in admitted or admitted in target:
            return True

    return False

def row_has_final_admit(row: dict, university: str) -> bool:
    for col, cell in row.items():
        if is_admit_column(col):
            for admitted in extract_universities(cell):
                if university_names_match(university, admitted):
                    return True
    return False

def row_has_any_final_admit(row: dict) -> bool:
    """Return True if the row has at least one final admitted university listed."""
    for col, cell in row.items():
        if is_admit_column(col) and str(cell).strip().lower() not in ["", "na", "n/a", "none"]:
            return True
    return False

COUNTRY_ALIASES = {
    "usa": ["usa", "us", "u s", "united states", "united states of america", "america"],
    "uk": ["uk", "u k", "united kingdom", "england", "scotland", "wales"],
    "canada": ["canada"],
    "singapore": ["singapore"],
    "india": ["india"],
}

def normalize_country(country: str) -> str:
    c = normalize(country)
    for canon, aliases in COUNTRY_ALIASES.items():
        if c == canon or c in aliases:
            return canon
    return c

def country_names_match(query_country: str, cell_value: str) -> bool:
    target = normalize_country(query_country)
    cell = normalize_country(cell_value)
    if not target or not cell:
        return False

    names = {target}
    for canon, aliases in COUNTRY_ALIASES.items():
        all_names = {normalize_country(canon), *[normalize_country(a) for a in aliases]}
        if target in all_names:
            names.update(all_names)

    return any(name == cell or name in cell or cell in name for name in names)

def is_countries_applied_column(col_name: str) -> bool:
    col = normalize(col_name)
    return "countr" in col and "appl" in col

def row_has_country_applied_to(row: dict, country: str) -> bool:
    for col, cell in row.items():
        if is_countries_applied_column(col):
            parts = [p.strip() for p in re.split(r"[;,/|]", str(cell)) if p.strip()]
            if any(country_names_match(country, part) for part in parts):
                return True
    return False

def is_generic_country_college_target(target: str) -> bool:
    """Detect phrases like 'US colleges' that are not university names."""
    t = normalize(target)
    generic_words = ["college", "colleges", "university", "universities", "school", "schools", "institutions"]
    country_words = []
    for canon, aliases in COUNTRY_ALIASES.items():
        country_words.extend([canon, *aliases])
    return any(cw in t for cw in country_words) and any(gw in t for gw in generic_words)

def query_mentions_country_colleges(query: str):
    """Return country canon such as 'usa' when query asks for US/UK/etc colleges."""
    q = normalize(query)
    generic = any(w in q for w in ["college", "colleges", "university", "universities", "schools"])
    if not generic:
        return None
    for canon, aliases in COUNTRY_ALIASES.items():
        if any(alias in q for alias in [canon, *aliases]):
            return canon
    return None

def sanitize_filters_for_country_college_query(user_query: str, filters: dict) -> dict:
    """Convert 'US colleges' style queries into country + any-admit filters.

    Without this, the LLM or regex can mistakenly treat 'US colleges' as a
    specific admitted university, which produces zero matches.
    """
    filters = dict(filters or {})
    country = query_mentions_country_colleges(user_query)

    for k in list(filters.keys()):
        k_norm = normalize(k).replace(" ", "_")
        is_admit_filter = k_norm in ["admitted", "admitted_univs", "admitted_university"] or is_admit_column(k)
        if is_admit_filter and is_generic_country_college_target(str(filters[k])):
            filters.pop(k, None)

    if country:
        filters["country_applied_to"] = country
        if re.search(r"\b(admit|admitted|accepted|got into|get into)\b", user_query.lower()):
            filters["require_any_final_admit"] = True

    return filters

def extract_admit_target_from_query(query: str):
    """
    Fallback extraction for natural questions like:
    - How many students got admitted to Cornell?
    - Tell me about students accepted to University of Michigan.
    - Who got into UC Berkeley?

    This helps when the LLM does not return a clean admitted_university filter.
    """
    q = clean_user_query(query)
    patterns = [
        r"(?:got|get)\s+into\s+(.+?)(?:[?.!,]|$)",
        r"(?:admitted|accepted)\s+(?:to|into|at)\s+(.+?)(?:[?.!,]|$)",
        r"(?:admit|admits|admission|admissions)\s+(?:to|into|at)\s+(.+?)(?:[?.!,]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            target = re.split(r"\s+from\s+|\s+at\s+|\s+in\s+", target, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if target and not is_generic_country_college_target(target):
                return target
            return None

    return None


# -------------------------------
# Conversation Memory Helpers
# -------------------------------
def is_followup_query(query: str) -> bool:
    """Detect short follow-up questions that rely on earlier context."""
    q = normalize(query)

    followup_starters = [
        "what about", "how about", "did they", "do they", "were they", "was the student",
        "what were their", "what was their", "their", "those students", "that student",
        "same students", "same student", "what did they", "where did they"
    ]

    followup_topics = [
        "sat", "act", "amc", "ap", "financial aid", "scholarship", "aid",
        "extracurricular", "extra curricular", "activities", "leadership", "courses",
        "projects", "research", "summer", "board", "scores", "grades", "lor", "ee"
    ]

    has_followup_language = any(starter in q for starter in followup_starters)
    has_topic = any(topic in q for topic in followup_topics)

    # A short topic-only query like "SAT scores?" or "Financial aid?" should also inherit context.
    short_query = len(q.split()) <= 7

    return has_followup_language or (short_query and has_topic)

def get_session_key(req: ChatRequest) -> str:
    """Prefer logged-in user memory; fall back to anonymous session_id."""
    user = get_user_by_token(req.auth_token)
    if user:
        return f"user:{user['id']}"
    return req.session_id.strip() if req.session_id and req.session_id.strip() else "default"

def get_authenticated_user_from_request(req: ChatRequest):
    return get_user_by_token(req.auth_token)

def get_memory(session_id: str) -> dict:
    return SESSION_MEMORY.get(session_id, {})

def update_memory(session_id: str, user_query: str, filters: dict, students: list):
    """Store the exact matched student profile group for follow-up questions."""
    if not students:
        return

    SESSION_MEMORY[session_id] = {
        "last_user_query": user_query,
        "last_filters": dict(filters or {}),
        "last_match_count": len(students),
        "last_students": [student_memory_keys(row) for row in students],
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_session_memory()

def apply_memory_to_filters(user_query: str, filters: dict, session_id: str) -> dict:
    """Merge previous filters into follow-up queries when the user omits context."""
    if not is_followup_query(user_query):
        return filters

    memory = get_memory(session_id)
    previous_filters = memory.get("last_filters", {})
    if not previous_filters:
        return filters

    merged = dict(previous_filters)
    merged.update(filters or {})  # Explicit current filters always win.
    return merged

def memory_context_sentence(session_id: str) -> str:
    memory = get_memory(session_id)
    previous_query = memory.get("last_user_query")
    previous_count = memory.get("last_match_count")

    if previous_query and previous_count is not None:
        return f'This appears to be a follow-up to the earlier query: "{previous_query}". Reuse that exact matched student group as the context.'
    return "No previous context is available."

def extract_student_id(row: dict):
    """Extract a stable student identifier from the row when available."""
    preferred_names = ["student id", "student_id", "id"]

    for col, val in row.items():
        col_norm = normalize(col).replace(" ", "_")
        if col_norm in {"student_id", "studentid"} and is_non_empty_cell(val):
            return str(val).strip()

    for col, val in row.items():
        col_norm = normalize(col)
        if any(name in col_norm for name in preferred_names) and is_non_empty_cell(val):
            return str(val).strip()

    return None

def row_signature(row: dict) -> str:
    """Fallback identifier when Student ID is absent."""
    school = extract_school_name(row)
    city = extract_city_of_graduation(row) or ""
    major = extract_major(row)
    admitted = []
    for col, cell in row.items():
        if is_admit_column(col):
            admitted.extend(extract_universities(cell))
    return normalize(f"{school}|{city}|{major}|{','.join(sorted(set(admitted)))}")

def student_memory_keys(row: dict):
    """Return both Student ID and fallback signature for profile memory."""
    return {
        "student_id": extract_student_id(row),
        "signature": row_signature(row),
    }

def filter_has_explicit_student_context(filters: dict) -> bool:
    """Detect whether the current query has its own selection context.

    If true, we should not blindly reuse the previous student group.
    """
    if not filters:
        return False

    context_keys = {
        "school", "school_name", "admitted", "admitted_univs", "admitted_university",
        "country_applied_to", "countries_applied_to", "country", "require_any_final_admit"
    }

    for k, v in filters.items():
        k_norm = normalize(k).replace(" ", "_")
        if k_norm in context_keys and is_non_empty_cell(v):
            return True
        if is_school_column(k) and is_non_empty_cell(v):
            return True
        if is_admit_column(k) and is_non_empty_cell(v):
            return True

    return False

def get_students_from_profile_memory(records: list, session_id: str):
    """Retrieve the exact previously matched student group from memory."""
    memory = get_memory(session_id)
    memory_students = memory.get("last_students", [])
    if not memory_students:
        return None

    remembered_ids = {s.get("student_id") for s in memory_students if s.get("student_id")}
    remembered_signatures = {s.get("signature") for s in memory_students if s.get("signature")}

    matched = []
    for row in records:
        sid = extract_student_id(row)
        sig = row_signature(row)
        if (sid and sid in remembered_ids) or (sig and sig in remembered_signatures):
            matched.append(row)

    return matched if matched else None

# -------------------------------
# Core Filter Engine (UNCHANGED)
# -------------------------------
def filter_students(records, filters):
    result = []

    key_map = {
        "school": "school_name",
        "school_name": "school_name",
        "admitted": "admitted_university",
        "admitted_univs": "admitted_university",
        "admitted_university": "admitted_university",
        "country_applied_to": "country_applied_to",
        "require_any_final_admit": "require_any_final_admit"
    }

    mapped_filters = {}
    for k, v in filters.items():
        k_norm = normalize(k).replace(" ", "_")
        if k_norm in key_map:
            mapped_filters[key_map[k_norm]] = v
        elif is_school_column(k):
            mapped_filters["school_name"] = v
        elif is_admit_column(k):
            mapped_filters["admitted_university"] = v

    for row in records:
        ok = True

        school = mapped_filters.get("school_name")
        if school and not row_has_school(row, school):
            ok = False

        admit = mapped_filters.get("admitted_university")
        if ok and admit and not row_has_final_admit(row, admit):
            ok = False

        country = mapped_filters.get("country_applied_to")
        if ok and country and not row_has_country_applied_to(row, country):
            ok = False

        if ok and mapped_filters.get("require_any_final_admit") and not row_has_any_final_admit(row):
            ok = False

        if ok:
            result.append(row)

    return result

# =========================================================
# PHASE-6.3 HELPERS
# =========================================================
def extract_school_name(row):
    for col, val in row.items():
        if is_school_column(col) and val:
            return val
    return "Unknown School"

def extract_city_of_graduation(row):
    """Extract city of graduation if the Google Sheet has a city/location column.

    This is intentionally conservative so we do not accidentally pick up
    unrelated city fields. If no city is found, return None and omit it
    from the student profile.
    """
    preferred_hints = [
        "city of graduation",
        "graduation city",
        "school city",
        "high school city",
        "city",
        "location"
    ]

    for hint in preferred_hints:
        for col, val in row.items():
            if hint in normalize(col) and val and str(val).strip().lower() not in ["na", "n/a", "none"]:
                return str(val).strip()

    return None

def extract_major(row):
    for col, val in row.items():
        if "major" in normalize(col) and val:
            return val
    return "Undeclared"

def extract_free_advice(row):
    for col, val in row.items():
        if "free" in normalize(col) and "advice" in normalize(col):
            if val and str(val).strip().lower() not in ["na", "n/a"]:
                return str(val).strip()
    return None


# =========================================================
# PHASE-4 FOLLOW-UP TOPIC / COLUMN SUPPORT
# =========================================================
EMPTY_CELL_VALUES = {"", "na", "n/a", "none", "null", "not applicable", "not specified"}

def is_non_empty_cell(val) -> bool:
    return val is not None and str(val).strip().lower() not in EMPTY_CELL_VALUES

def get_column_value(row: dict, candidate_names):
    """Return a value from a row using exact/fuzzy column-name matching."""
    normalized_candidates = [normalize(name) for name in candidate_names]

    # Prefer exact normalized column matches first.
    for col, val in row.items():
        if normalize(col) in normalized_candidates and is_non_empty_cell(val):
            return str(val).strip()

    # Then allow conservative substring matching.
    for col, val in row.items():
        col_norm = normalize(col)
        for cand in normalized_candidates:
            if (cand in col_norm or col_norm in cand) and is_non_empty_cell(val):
                return str(val).strip()

    return None

FOLLOWUP_FIELD_RULES = [
    {
        "label": "Academic Extra-curriculars",
        "triggers": [
            "course", "courses", "extracurricular", "extra curricular", "summer program",
            "summer programs", "academic program", "academic programs", "academic project",
            "academic projects", "research", "project", "projects", "activity", "activities"
        ],
        "columns": ["Academic Extra-curriculars", "Academic Extracurriculars", "Academic Extra Curriculars"],
    },
    {
        "label": "Non-Academic Extra-curriculars",
        "triggers": [
            "extracurricular", "extra curricular", "non academic", "non-academic",
            "non-academic activity", "non academic activity", "activity", "activities", "sports", "music", "art", "volunteer"
        ],
        "columns": ["Non-Academic Extra-curriculars", "Non Academic Extra-curriculars", "Non-Academic Extracurriculars"],
    },
    {
        "label": "Financial Aid",
        "triggers": ["financial aid", "scholarship", "scholarships", "aid", "funding"],
        "columns": ["Financial Aid"],
    },
    {
        "label": "Leadership Roles Held",
        "triggers": ["lead role", "leadership", "leadership role", "leader", "captain", "president", "founder"],
        "columns": ["Leadership roles held", "Leadership Roles Held"],
    },
    {
        "label": "SAT Total Score",
        "triggers": ["sat", "sat score", "sat total"],
        "columns": ["SAT Total score", "SAT Total Score"],
    },
    {
        "label": "ACT Score",
        "triggers": ["act", "act score"],
        "columns": ["ACT Score"],
    },
    {
        "label": "AMC-10 Taken",
        "triggers": ["amc-10 taken", "amc 10 taken", "took amc-10", "took amc 10"],
        "columns": ["AMC-10 taken", "AMC 10 taken"],
    },
    {
        "label": "AMC-12 Taken",
        "triggers": ["amc-12 taken", "amc 12 taken", "took amc-12", "took amc 12"],
        "columns": ["AMC-12 taken", "AMC 12 taken"],
    },
    {
        "label": "AMC-10 Score",
        "triggers": ["amc-10 score", "amc 10 score"],
        "columns": ["AMC-10 Score", "AMC 10 Score"],
    },
    {
        "label": "AMC-12 Score",
        "triggers": ["amc-12 score", "amc 12 score"],
        "columns": ["AMC-12 Score", "AMC 12 Score"],
    },
    {
        "label": "AP Courses",
        "triggers": ["ap course", "ap courses", "advanced placement", "ap exam", "ap exams"],
        "columns": ["AP Courses"],
    },
    {
        "label": "10th Board",
        "triggers": ["10th board", "grade 10 board", "class 10 board", "tenth board"],
        "columns": ["10th Board"],
    },
    {
        "label": "12th Board",
        "triggers": ["12th board", "grade 12 board", "class 12 board", "twelfth board"],
        "columns": ["12th Board"],
    },
    {
        "label": "9th Grade Scores",
        "triggers": ["9th grade score", "9th grade scores", "grade 9 score", "grade 9 scores", "class 9 score"],
        "columns": ["9th grade scores", "9th Grade Scores"],
    },
    {
        "label": "10th Grade Scores",
        "triggers": ["10th grade score", "10th grade scores", "grade 10 score", "grade 10 scores", "class 10 score"],
        "columns": ["10th grade scores", "10th Grade Scores"],
    },
    {
        "label": "11th Grade Overall Score",
        "triggers": ["11th grade overall", "grade 11 overall", "class 11 overall", "11th overall"],
        "columns": ["11th grade overall score", "11th Grade Overall Score"],
    },
    {
        "label": "12th Grade Scores",
        "triggers": ["12th grade score", "12th grade scores", "grade 12 score", "grade 12 scores", "class 12 score"],
        "columns": ["12th grade scores", "12th Grade Scores"],
    },
    {
        "label": "12th Grade Overall Score",
        "triggers": ["12th grade overall", "grade 12 overall", "class 12 overall", "12th overall"],
        "columns": ["12th grade overall score", "12th Grade Overall Score"],
    },
]

def requested_followup_fields(user_query: str):
    """Identify which database fields should be included for this query."""
    q = normalize(user_query)
    requested = []
    seen_labels = set()

    # Special case: a generic AMC query should include both taken and score fields.
    if "amc" in q and "score" not in q and "taken" not in q:
        for label in ["AMC-10 Taken", "AMC-10 Score", "AMC-12 Taken", "AMC-12 Score"]:
            for rule in FOLLOWUP_FIELD_RULES:
                if rule["label"] == label and label not in seen_labels:
                    requested.append(rule)
                    seen_labels.add(label)

    for rule in FOLLOWUP_FIELD_RULES:
        if rule["label"] in seen_labels:
            continue
        if any(trigger in q for trigger in rule["triggers"]):
            requested.append(rule)
            seen_labels.add(rule["label"])

    return requested

def format_requested_followup_details(row: dict, requested_rules) -> str:
    """Build a concise block of requested profile details for the LLM context."""
    lines = []
    for rule in requested_rules:
        val = get_column_value(row, rule["columns"])
        if val:
            lines.append(f'{rule["label"]}: {val}')
    return "\n".join(lines)



# =========================================================
# PHASE-6.5 PATTERN ANALYTICS SUPPORT
# =========================================================
def split_cell_values(cell: str):
    """Split a multi-value spreadsheet cell into clean values."""
    if not is_non_empty_cell(cell):
        return []

    text = str(cell).strip()
    # Keep this conservative: common separators in form responses.
    parts = re.split(r"[;|\n]+|,(?=\s*[A-Za-z0-9])", text)
    cleaned = []
    for part in parts:
        item = part.strip(" \t-•")
        if item and item.lower() not in EMPTY_CELL_VALUES:
            cleaned.append(item)
    return cleaned


def add_count(counter: dict, value: str):
    if not is_non_empty_cell(value):
        return
    key = str(value).strip()
    counter[key] = counter.get(key, 0) + 1


def top_counts(counter: dict, limit: int = 5):
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower()))[:limit]


def extract_number(value):
    """Extract the first plausible number from a score field."""
    if not is_non_empty_cell(value):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def format_numeric_pattern(label: str, values: list):
    if not values:
        return None
    values_sorted = sorted(values)
    avg = sum(values_sorted) / len(values_sorted)
    low = values_sorted[0]
    high = values_sorted[-1]

    def fmt(x):
        return str(int(x)) if float(x).is_integer() else f"{x:.1f}"

    if len(values_sorted) == 1:
        return f"{label}: one specified value, {fmt(values_sorted[0])}"
    return f"{label}: {len(values_sorted)} specified values; range {fmt(low)}–{fmt(high)}; average about {fmt(avg)}"


def build_pattern_analytics(students: list, requested_rules: list) -> str:
    """Create deterministic, data-backed pattern notes for the LLM.

    The LLM may phrase these nicely, but the counts and ranges are computed here
    so Mentorly does not invent analytics.
    """
    count = len(students)
    if count == 0:
        return "No matching students, so no pattern analytics are available."

    lines = [f"Matched student count: {count}"]

    school_counts = {}
    city_counts = {}
    major_counts = {}
    admit_counts = {}

    for row in students:
        add_count(school_counts, extract_school_name(row))
        city = extract_city_of_graduation(row)
        if city:
            add_count(city_counts, city)
        add_count(major_counts, extract_major(row))

        for col, cell in row.items():
            if is_admit_column(col):
                for university in extract_universities(cell):
                    add_count(admit_counts, university)

    if count >= 2:
        if top_counts(school_counts):
            lines.append("High schools represented: " + "; ".join(f"{k} ({v})" for k, v in top_counts(school_counts)))
        if top_counts(city_counts):
            lines.append("Cities of graduation represented: " + "; ".join(f"{k} ({v})" for k, v in top_counts(city_counts)))
        if top_counts(major_counts):
            lines.append("Intended major pattern: " + "; ".join(f"{k} ({v})" for k, v in top_counts(major_counts)))
        if top_counts(admit_counts):
            lines.append("Admitted university pattern: " + "; ".join(f"{k} ({v})" for k, v in top_counts(admit_counts)))
    else:
        lines.append("Only one matching student is available, so discuss this as a single profile rather than a broad trend.")

    # Add pattern summaries for the specific requested follow-up columns.
    for rule in requested_rules:
        label = rule["label"]
        raw_values = []
        numeric_values = []
        specified_count = 0

        for row in students:
            val = get_column_value(row, rule["columns"])
            if val:
                specified_count += 1
                raw_values.append(val)
                num = extract_number(val)
                if num is not None:
                    numeric_values.append(num)

        if specified_count == 0:
            lines.append(f"{label}: not specified for the matching student group.")
            continue

        numeric_line = None
        if any(score_word in normalize(label) for score_word in ["score", "sat", "act"]):
            numeric_line = format_numeric_pattern(label, numeric_values)

        if numeric_line:
            lines.append(numeric_line)
            continue

        value_counts = {}
        for raw in raw_values:
            parts = split_cell_values(raw)
            if not parts:
                parts = [raw]
            for part in parts:
                add_count(value_counts, part)

        if count >= 2:
            lines.append(
                f"{label}: specified for {specified_count}/{count} matching students; "
                + "; ".join(f"{k} ({v})" for k, v in top_counts(value_counts, limit=6))
            )
        else:
            lines.append(f"{label}: {raw_values[0]}")

    return "\n".join(lines)


# =========================================================
# PHASE-8 ADVANCED FEATURES: ADVISORY, SIMILARITY, INSIGHTS, DASHBOARD
# =========================================================
def get_all_admitted_universities(row: dict):
    admitted = []
    for col, cell in row.items():
        if is_admit_column(col):
            admitted.extend(extract_universities(cell))
    return sorted(set([u for u in admitted if u]))

def is_university_applied_column(col_name: str) -> bool:
    col = normalize(col_name)
    return "appl" in col and "countr" not in col and any(k in col for k in ["univ", "college", "school"])

def row_has_university_in_applied_or_admitted(row: dict, university: str) -> bool:
    if row_has_final_admit(row, university):
        return True
    for col, cell in row.items():
        if is_university_applied_column(col):
            for uni in extract_universities(cell):
                if university_names_match(university, uni):
                    return True
    return False

def extract_university_mentioned_in_query(query: str, records=None):
    target = extract_admit_target_from_query(query)
    if target:
        return target

    q = clean_user_query(query)
    patterns = [
        r"(?:insights|applicants|students|profile|profiles|advice|guidance)\s+(?:for|about|to|at)\s+(.+?)(?:[?.!,]|$)",
        r"(?:for|about|to|at)\s+([A-Z][A-Za-z& .'-]+?)(?:\s+(?:applicants|admits|students|admissions))?(?:[?.!,]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r"\b(applicants|admits|students|admissions|university|college insights)$", "", candidate, flags=re.IGNORECASE).strip()
            if candidate and len(candidate) > 1 and not is_generic_country_college_target(candidate):
                return candidate

    # Fallback: scan known admitted universities from the database and find one named in the query.
    if records:
        q_norm = normalize(q)
        known = set()
        for row in records:
            known.update(get_all_admitted_universities(row))
        for uni in sorted(known, key=len, reverse=True):
            if uni and (uni in q_norm or q_norm in uni):
                return uni
    return None

def build_student_profile_block(row: dict, index: int = 1, requested_rules=None) -> str:
    requested_rules = requested_rules or []
    school = extract_school_name(row)
    city = extract_city_of_graduation(row)
    major = extract_major(row)
    admitted = get_all_admitted_universities(row)
    advice = extract_free_advice(row) or "No advice provided."
    requested_details = format_requested_followup_details(row, requested_rules)
    requested_details_block = f"\nRequested Details:\n{requested_details}" if requested_details else ""
    student_id = extract_student_id(row)
    id_line = f"Student ID: {student_id}\n" if student_id else ""
    return f"""
Student {index}
{id_line}High School: {school}
City of Graduation: {city if city else "Not specified"}
Intended Major: {major}
Admitted Universities: {", ".join(admitted) if admitted else "Not specified"}
Advice: {advice}{requested_details_block}
""".strip()

def answer_with_llm(intent: str, user_query: str, students: list, prompt_instructions: str, requested_rules=None, session_id="default"):
    requested_rules = requested_rules or requested_followup_fields(user_query)
    pattern_analytics = build_pattern_analytics(students, requested_rules)
    blocks = [build_student_profile_block(row, i, requested_rules) for i, row in enumerate(students[:12], start=1)]
    prompt = f"""
You are Mentorly, a warm, fluent, and thoughtful college counseling assistant.

Answer the user using ONLY the student records and deterministic analytics below.

User Question:
"{user_query}"

Intent:
{intent}

Total matching students available: {len(students)}

Deterministic Pattern Analytics:
{pattern_analytics}

Student Records:
{chr(10).join(blocks) if blocks else "No matching student records."}

Instructions for this answer:
{prompt_instructions}

Global rules:
- Do not invent universities, scores, schools, activities, financial aid, teachers, outcomes, or other facts.
- Do not use outside knowledge.
- Numerical claims must come from Deterministic Pattern Analytics or the student records.
- Be warm, concrete, and counselor-like.
- Use plain text section titles only. Do not use #, ##, ###, or **.
- If the data is thin, say that gently in the normal answer, not as a separate Limitations section.
- End with one helpful, self-contained follow-up question.
"""
    try:
        llm_response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.45,
        )
        content = clean_assistant_markdown(llm_response.choices[0].message.content.strip())
        return {"intent": intent, "assistant_answer": content}
    except Exception:
        return handle_analytics_response(user_query, students)

def extract_filters_for_query(user_query: str):
    all_columns = sheet.row_values(1)
    normalized_columns = [normalize(col).replace(" ", "_") for col in all_columns]
    prompt = f"""
Convert the user query into JSON.
Allowed keys:
{', '.join(normalized_columns)}

User query:
"{user_query}"
"""
    try:
        raw = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        filters = json.loads(match.group()) if match else {}
    except Exception:
        filters = {}
    filters = sanitize_filters_for_country_college_query(user_query, filters)
    if re.search(r"\b(admit|admitted|accepted|got into|get into)\b", user_query.lower()):
        mapped = {}
        for k, v in filters.items():
            k_norm = normalize(k).replace(" ", "_")
            if k_norm in {"admitted", "admitted_univs", "admitted_university"} or is_admit_column(k):
                mapped["admitted_university"] = v
        if "admitted_university" not in mapped:
            admit_target = extract_admit_target_from_query(user_query)
            if admit_target:
                filters["admitted_university"] = admit_target
    if re.search(r"\bschool\b", user_query.lower()):
        has_school = any(normalize(k).replace(" ", "_") in {"school", "school_name"} or is_school_column(k) for k in filters)
        if not has_school:
            match = re.search(r"from\s+(.+?school)", user_query, re.IGNORECASE)
            if match:
                filters["school_name"] = match.group(1).strip()
    return sanitize_filters_for_country_college_query(user_query, filters)

def handle_advisory_flow(user_query: str, records: list, session_id: str):
    filters = extract_filters_for_query(user_query)
    university = extract_university_mentioned_in_query(user_query, records)
    if university and "admitted_university" not in {normalize(k).replace(" ", "_"): v for k, v in filters.items()}:
        filters["admitted_university"] = university
    students = filter_students(records, filters) if filters else records
    if not students and university:
        students = [row for row in records if row_has_university_in_applied_or_admitted(row, university)]
    students = students[:12]
    update_memory(session_id, user_query, filters, students)
    instructions = """
This is an advisory answer. Give practical guidance grounded in similar or relevant student records.
Structure:
1. Start with a direct, reassuring answer.
2. Summarize the most relevant student examples.
3. Give specific advice grounded in those examples: academics, testing, activities, leadership, projects/research, recommendations, and financial aid only if those appear in the records.
4. Do not imply guaranteed admission or use outside admissions knowledge.
5. If no strong matching records exist, say what the database can and cannot support, then give cautious guidance based only on available rows.
"""
    return answer_with_llm("advisory", user_query, students, instructions, requested_followup_fields(user_query), session_id)

def parse_user_profile_from_query(user_query: str) -> dict:
    q = user_query
    profile = {}
    sat = re.search(r"\bSAT\D{0,12}(\d{3,4})\b|\b(\d{3,4})\D{0,12}SAT\b", q, re.IGNORECASE)
    if sat:
        profile["sat"] = int(next(g for g in sat.groups() if g))
    act = re.search(r"\bACT\D{0,12}(\d{1,2})\b|\b(\d{1,2})\D{0,12}ACT\b", q, re.IGNORECASE)
    if act:
        profile["act"] = int(next(g for g in act.groups() if g))
    major_match = re.search(r"(?:major(?:ing)? in|interested in|for)\s+([A-Za-z /&-]{2,60})(?:[.,;]|$)", q, re.IGNORECASE)
    if major_match:
        profile["major"] = major_match.group(1).strip()
    country = query_mentions_country_colleges(q)
    if country:
        profile["country"] = country
    return profile

def score_similarity(row: dict, profile: dict, query: str) -> tuple:
    score = 0
    reasons = []
    q_norm = normalize(query)

    major = extract_major(row)
    if profile.get("major") and profile["major"]:
        if normalize(profile["major"]) in normalize(major) or normalize(major) in normalize(profile["major"]):
            score += 30
            reasons.append(f"similar intended major ({major})")
    elif major and any(tok in normalize(major) for tok in q_norm.split() if len(tok) > 3):
        score += 15
        reasons.append(f"related major ({major})")

    sat_val = get_column_value(row, ["SAT Total score", "SAT Total Score"])
    sat_num = extract_number(sat_val) if sat_val else None
    if profile.get("sat") and sat_num:
        diff = abs(profile["sat"] - sat_num)
        if diff <= 30:
            score += 30
        elif diff <= 80:
            score += 20
        elif diff <= 150:
            score += 10
        if diff <= 150:
            reasons.append(f"SAT is close ({sat_val})")

    act_val = get_column_value(row, ["ACT Score"])
    act_num = extract_number(act_val) if act_val else None
    if profile.get("act") and act_num:
        diff = abs(profile["act"] - act_num)
        if diff <= 1:
            score += 25
        elif diff <= 3:
            score += 15
        if diff <= 3:
            reasons.append(f"ACT is close ({act_val})")

    if profile.get("country") and row_has_country_applied_to(row, profile["country"]):
        score += 15
        reasons.append(f"applied to {profile['country'].upper()} colleges")

    # Keyword overlap with activities/advice gives a lightweight profile match.
    searchable = " ".join(str(v) for v in row.values())
    overlap = 0
    for token in set(q_norm.split()):
        if len(token) >= 5 and token in normalize(searchable):
            overlap += 1
    if overlap:
        score += min(20, overlap * 3)
        reasons.append("overlapping activities/interests")

    if row_has_any_final_admit(row):
        score += 5
    return score, reasons

def handle_similar_student_search(user_query: str, records: list, session_id: str):
    profile = parse_user_profile_from_query(user_query)
    scored = []
    for row in records:
        score, reasons = score_similarity(row, profile, user_query)
        if score > 0:
            scored.append((score, reasons, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    students = [row for _, _, row in scored[:5]]
    if not students:
        students = records[:5]
    for idx, (score, reasons, row) in enumerate(scored[:5]):
        row["_similarity_score"] = score
        row["_similarity_reasons"] = "; ".join(reasons) if reasons else "general profile overlap"
    update_memory(session_id, user_query, {"similarity_search": True, "profile": profile}, students)
    instructions = """
This is a similar-student search. Explain which database students look closest to the user's profile.
Structure:
1. Start with a direct answer that these are the closest matching profiles found.
2. For each student, mention high school, city, intended major, admits, and why the profile seems similar when available.
3. Use SAT, ACT, activities, major, country, and other fields only when present in records.
4. Give practical guidance based on the similarities.
5. Do not call this a perfect match; say it is a directional match from the database.
"""
    return answer_with_llm("similarity", user_query, students, instructions, requested_followup_fields(user_query), session_id)

def handle_university_insights(user_query: str, records: list, session_id: str):
    university = extract_university_mentioned_in_query(user_query, records)
    if not university:
        return {"intent": "university_insights", "assistant_answer": "Please mention the university name so I can build the university insights view."}
    students = [row for row in records if row_has_university_in_applied_or_admitted(row, university)]
    admitted_students = [row for row in students if row_has_final_admit(row, university)]
    context_students = admitted_students or students
    filters = {"admitted_university": university} if admitted_students else {"university_interest": university}
    update_memory(session_id, user_query, filters, context_students)
    requested = requested_followup_fields(user_query)
    # Add important fields for an insights page even if not explicitly requested.
    for label in ["SAT Total Score", "ACT Score", "AP Courses", "Academic Extra-curriculars", "Non-Academic Extra-curriculars", "Financial Aid", "Leadership Roles Held"]:
        for rule in FOLLOWUP_FIELD_RULES:
            if rule["label"] == label and all(r["label"] != label for r in requested):
                requested.append(rule)
    instructions = f"""
This is a University Insights Page for {university}.
Structure:
1. Start with a short overview: how many relevant profiles were found and how many were admitted when available.
2. Summarize admitted-student profiles first if any exist.
3. Include patterns in majors, schools/cities, scores, activities, leadership, financial aid, and advice only when those fields are present.
4. Keep it grounded and avoid outside admissions commentary.
5. Make it useful as a mini admissions intelligence page for this university.
"""
    return answer_with_llm("university_insights", user_query, context_students, instructions, requested, session_id)

def build_dashboard_payload(records: list) -> dict:
    total = len(records)
    admitted_rows = [r for r in records if row_has_any_final_admit(r)]
    high_schools = Counter(extract_school_name(r) for r in records if extract_school_name(r) != "Unknown School")
    cities = Counter(extract_city_of_graduation(r) for r in records if extract_city_of_graduation(r))
    majors = Counter(extract_major(r) for r in records if extract_major(r) != "Undeclared")
    admitted_unis = Counter()
    for r in records:
        admitted_unis.update(get_all_admitted_universities(r))
    sat_values = []
    for r in records:
        sat = get_column_value(r, ["SAT Total score", "SAT Total Score"])
        n = extract_number(sat) if sat else None
        if n:
            sat_values.append(n)
    return {
        "total_students": total,
        "students_with_final_admits": len(admitted_rows),
        "top_high_schools": high_schools.most_common(10),
        "top_cities": cities.most_common(10),
        "top_intended_majors": majors.most_common(10),
        "top_admitted_universities": admitted_unis.most_common(15),
        "sat_summary": {
            "count": len(sat_values),
            "min": min(sat_values) if sat_values else None,
            "max": max(sat_values) if sat_values else None,
            "average": round(sum(sat_values) / len(sat_values), 1) if sat_values else None,
        },
    }

def format_dashboard_text(payload: dict) -> str:
    def fmt_pairs(pairs):
        return "\n".join([f"- {name}: {count}" for name, count in pairs]) if pairs else "Not enough data specified."
    sat = payload["sat_summary"]
    sat_text = "Not enough SAT data specified."
    if sat["count"]:
        sat_text = f"{sat['count']} scores available; range {sat['min']}–{sat['max']}; average {sat['average']}."
    return f"""
Dashboard Summary

Total students in database: {payload['total_students']}
Students with final admits listed: {payload['students_with_final_admits']}

Top High Schools
{fmt_pairs(payload['top_high_schools'])}

Top Cities
{fmt_pairs(payload['top_cities'])}

Top Intended Majors
{fmt_pairs(payload['top_intended_majors'])}

Top Admitted Universities
{fmt_pairs(payload['top_admitted_universities'])}

SAT Summary
{sat_text}
""".strip()

def handle_dashboard_query(user_query: str, records: list, session_id: str):
    payload = build_dashboard_payload(records)
    return {"intent": "dashboard", "assistant_answer": format_dashboard_text(payload), "dashboard": payload}

def clean_assistant_markdown(text: str) -> str:
    """Remove markdown artifacts from the LLM response before sending to the frontend."""
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?mi)^\s*Direct Answer\s*:?\s*\n+", "", text)
    text = re.sub(r"(?mi)^\s*What the Records Show\s*:?\s*\n+", "", text)
    return text.strip()

# =========================================================
# PHASE-6.3 BASE ANALYTICS RESPONSE (FIXED)
# =========================================================
def handle_analytics_response(query, students):
    count = len(students)

    if count == 0:
        return {
            "intent": "analytics",
            "assistant_answer": "No matching students were found for this query."
        }

    intro = (
        f"There is {count} student who matches your query."
        if count == 1
        else f"There are {count} students who match your query."
    )

    lines = []

    for row in students:
        school = extract_school_name(row)
        city = extract_city_of_graduation(row)
        major = extract_major(row)

        admitted_univs = []
        for col, cell in row.items():
            if is_admit_column(col):
                admitted_univs.extend(extract_universities(cell))

        admitted_text = ", ".join(set(admitted_univs)) if admitted_univs else "University not specified"
        advice = extract_free_advice(row)

        city_line = f"City of Graduation: {city}\n" if city else ""

        entry = (
            f"High School: {school}\n"
            f"{city_line}"
            f"Intended Major: {major}\n"
            f"Admitted University: {admitted_text}"
        )

        if advice:
            entry += f"\nAdvice: {advice}"

        lines.append(entry)

    response = intro + "\n\n" + "\n\n".join(lines)

    return {
        "intent": "analytics",
        "assistant_answer": response.strip()
    }

# =========================================================
# PHASE-6.3 LLM NLG LAYER (FIXED)
# =========================================================
def generate_nlg_response(user_query, students, base_response, session_id="default"):
    try:
        count = len(students)
        followup_rules = requested_followup_fields(user_query)
        inherited_context_note = memory_context_sentence(session_id) if is_followup_query(user_query) else "This is not a follow-up query."
        requested_field_labels = [rule["label"] for rule in followup_rules]
        pattern_analytics = build_pattern_analytics(students, followup_rules)

        student_blocks = []
        for i, row in enumerate(students, start=1):
            school = extract_school_name(row)
            city = extract_city_of_graduation(row)
            major = extract_major(row)

            admitted = []
            for col, cell in row.items():
                if is_admit_column(col):
                    admitted.extend(extract_universities(cell))

            advice = extract_free_advice(row) or "No advice provided."
            requested_details = format_requested_followup_details(row, followup_rules)
            requested_details_block = (
                f"\n\nRequested Follow-up Details:\n{requested_details}"
                if requested_details
                else ""
            )

            student_blocks.append(
                f"""
Student {i}

High School: {school}

City of Graduation: {city if city else "Not specified"}

Intended Major: {major}

Admitted Universities: {", ".join(set(admitted)) if admitted else "Not specified"}

Advice: {advice}{requested_details_block}
""".strip()
            )

        prompt = f"""
You are Mentorly, a warm, fluent, and thoughtful college counseling assistant.

Your task is to answer the user's question using ONLY the student records provided below.

User Question:
"{user_query}"

Total matching students found: {count}

Conversation Context:
{inherited_context_note}

Requested follow-up field categories detected from the user question:
{", ".join(requested_field_labels) if requested_field_labels else "None"}

Deterministic Pattern Analytics:
{pattern_analytics}

Student Records:
{chr(10).join(student_blocks)}

Write a polished, helpful response with this structure:

1. Start immediately with a direct answer in a normal sentence. Do not use the heading "Direct Answer".
2. In the next paragraph, directly describe the matching student profile(s). Do not use the heading "What the Records Show".
3. If the user asks about SAT, ACT, AMC, AP courses, academics, boards, grades, extracurricular activities, summer programs, academic programs, projects, leadership, financial aid, or scholarships, answer using the "Requested Follow-up Details" provided in each student record.
4. If teacher names are explicitly mentioned in the Advice text, include a plain-text section titled "Teacher and Mentor Support".
5. Include a plain-text section titled "Patterns and Takeaways" when there is enough relevant information. Use the Deterministic Pattern Analytics section for counts, ranges, averages, and repeated themes. Do not create numerical patterns that are not listed there.
6. Include a plain-text section titled "Practical Guidance".
7. End with one helpful, self-contained follow-up question the user could ask next. The follow-up question should include the university, school, or student context where possible, not vague phrases like "this student".

Important rules:
- Do not invent universities, scores, schools, activities, outcomes, teachers, EE supervisors, or other facts.
- Do not use outside knowledge.
- You may synthesize patterns from the provided records, but numerical pattern claims must come from the Deterministic Pattern Analytics section.
- You may rephrase student advice to improve clarity, but do not change its meaning.
- If the user asks about a specific data category, such as SAT, AP courses, extracurriculars, leadership, financial aid, AMC, ACT, boards, or grade scores, prioritize the corresponding Requested Follow-up Details and do not answer generically when those details are available.
- If requested follow-up details are unavailable for a matching student, say that the field is not specified for that student rather than guessing.
- If the available data is thin, say so clearly in the normal answer, but do not create a separate "Limitations" section.
- Be warm, conversational, and encouraging.
- Write like an experienced college counselor.
- Do not use the section title "What the Records Show". After the first sentence, immediately describe the student profile(s).
- Avoid phrases such as "the records show", "the dataset indicates", and "the student records indicate".
- Use student-centric phrasing such as "The student graduated from...", "The student intends to major in...", and "The student was admitted to...".
- If only one student matches, discuss that student specifically. Do not generalize from one student by saying "students aiming for" or "this record suggests that students". Use phrasing like "The student is aiming for...".
- Use plain text section titles only, such as "Teacher and Mentor Support", "Patterns and Takeaways", "Practical Guidance", and "Suggested Next Question".
- Do not use markdown formatting. Do not use #, ##, ###, or ** anywhere in the response.
- When discussing individual students, always mention the student's High School, City of Graduation if available, Intended Major, and Admitted Universities if those fields are available.
- From the Advice text, extract teacher or mentor names only when they are explicitly stated. If a teacher helped with LOR, Extended Essay, projects, or research, mention that relationship clearly. If no teacher/mentor names are explicitly present, omit the "Teacher and Mentor Support" section entirely. Do not invent teacher names or roles.
- Focus on actionable insights rather than simply listing students.
- If there are multiple matching students, compare them and call out repeated schools, cities, majors, admitted universities, score ranges, financial aid patterns, extracurricular patterns, leadership patterns, and other requested fields when those analytics are provided.
- If there is only one matching student, avoid broad pattern claims and describe the single profile directly.
"""

        llm_response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        content = clean_assistant_markdown(llm_response.choices[0].message.content.strip())
        if not content:
            raise ValueError("Empty LLM response")

        return {
            "intent": "analytics",
            "assistant_answer": content
        }

    except Exception:
        return base_response

# -------------------------------
# API
# -------------------------------
@app.post("/signup")
async def signup(req: SignupRequest):
    username = req.username.strip()
    email = req.email.strip().lower()
    password = req.password

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters long.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    now = datetime.utcnow().isoformat() + "Z"
    password_hash = hash_password(password)

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (username, email, password_hash, login_count, created_at, last_login)
            VALUES (?, ?, ?, 0, ?, NULL)
            """,
            (username, email, password_hash, now),
        )
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username or email already exists.")

    token = create_auth_session(user_id)

    return {
        "success": True,
        "message": "Account created successfully.",
        "user_id": user_id,
        "username": username,
        "email": email,
        "auth_token": token,
    }

@app.post("/login")
async def login(req: LoginRequest):
    username = req.username.strip()
    password = req.password

    user = get_user_by_username_or_email(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")

    now = datetime.utcnow().isoformat() + "Z"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE users
        SET login_count = COALESCE(login_count, 0) + 1,
            last_login = ?
        WHERE id = ?
        """,
        (now, user["id"]),
    )
    conn.commit()
    conn.close()

    # Re-read updated user record.
    user = get_user_by_username_or_email(username)
    token = create_auth_session(user["id"])

    return {
        "success": True,
        "message": "Login successful.",
        "user_id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "login_count": user["login_count"],
        "last_login": user["last_login"],
        "auth_token": token,
    }

@app.post("/logout")
async def logout(req: AuthRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM auth_sessions WHERE token = ?", (req.auth_token,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Logged out successfully."}

@app.post("/me")
async def me(req: AuthRequest):
    user = get_user_by_token(req.auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired auth token.")

    return {
        "user_id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "login_count": user["login_count"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
    }

@app.get("/admin/users")
async def admin_users():
    """Basic user stats endpoint. Protect this later before public launch."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, email, login_count, created_at, last_login
        FROM users
        ORDER BY id DESC
        """
    )
    users = [dict(row) for row in cur.fetchall()]
    cur.execute("SELECT COUNT(*) AS cnt FROM chat_history")
    chat_count = cur.fetchone()["cnt"]
    conn.close()
    return {
        "total_users": len(users),
        "total_chat_questions": chat_count,
        "users": users,
    }


def with_chat_history(response: dict, user_id: Optional[int], session_id: str, question: str):
    try:
        answer = response.get("assistant_answer", "")
        record_chat_history(user_id, session_id, question, answer)
    except Exception:
        pass
    return response

@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    user_query = clean_user_query(req.message)
    authenticated_user = get_authenticated_user_from_request(req)
    session_id = get_session_key(req)
    intent = classify_intent(user_query)
    records = sheet.get_all_records()

    # New feature routes. These are additive and do not disturb the existing analytics flow.
    user_id = authenticated_user["id"] if authenticated_user else None

    if intent == "dashboard":
        response = handle_dashboard_query(user_query, records, session_id)
        return with_chat_history(response, user_id, session_id, user_query)

    if intent == "similarity":
        response = handle_similar_student_search(user_query, records, session_id)
        return with_chat_history(response, user_id, session_id, user_query)

    if intent == "university_insights":
        response = handle_university_insights(user_query, records, session_id)
        return with_chat_history(response, user_id, session_id, user_query)

    if intent == "advisory":
        response = handle_advisory_flow(user_query, records, session_id)
        return with_chat_history(response, user_id, session_id, user_query)

    filters = extract_filters_for_query(user_query)

    explicit_context_in_current_query = filter_has_explicit_student_context(filters)

    # Student profile memory:
    # For follow-ups like "What about their SAT scores?" or "Did they get financial aid?",
    # reuse the exact student group from the previous turn instead of asking the LLM
    # to reconstruct the same filters.
    memory_students = None
    if is_followup_query(user_query) and not explicit_context_in_current_query:
        memory_students = get_students_from_profile_memory(records, session_id)

    if memory_students is not None:
        students = memory_students
    else:
        # If this is a follow-up with no exact student group available, inherit
        # the previous filters as a fallback.
        filters = apply_memory_to_filters(user_query, filters, session_id)
        students = filter_students(records, filters)

    # Save the exact matched student group for the next turn.
    update_memory(session_id, user_query, filters, students)

    base_response = handle_analytics_response(user_query, students)
    response = generate_nlg_response(user_query, students, base_response, session_id=session_id)
    return with_chat_history(response, user_id, session_id, user_query)

@app.get("/dashboard")
async def dashboard():
    records = sheet.get_all_records()
    return build_dashboard_payload(records)

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    return get_memory(session_id)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in SESSION_MEMORY:
        SESSION_MEMORY.pop(session_id, None)
        save_session_memory()
    return {"status": "cleared", "session_id": session_id}
