from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import gspread
from google.oauth2.service_account import Credentials
import os, json, re

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
# Lightweight Conversation Memory
# -------------------------------
# Render instances are stateless across restarts, so this memory is intentionally
# lightweight and in-process. It works well for short chat sessions while the
# same backend instance is running. For production-grade memory, replace this
# with Redis, a database, or frontend-managed conversation state.
SESSION_MEMORY = {}


# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

# -------------------------------
# Intent Classification
# -------------------------------
def classify_intent(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["how many", "count", "number of"]):
        return "analytics"
    if any(k in q for k in ["advice", "guidance", "counsel"]):
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
    return req.session_id.strip() if req.session_id and req.session_id.strip() else "default"

def get_memory(session_id: str) -> dict:
    return SESSION_MEMORY.get(session_id, {})

def update_memory(session_id: str, user_query: str, filters: dict, students: list):
    """Store the last useful filter context so follow-up questions can reuse it."""
    if not filters:
        return

    SESSION_MEMORY[session_id] = {
        "last_user_query": user_query,
        "last_filters": dict(filters),
        "last_match_count": len(students),
    }

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
        return f'This appears to be a follow-up to the earlier query: "{previous_query}". Reuse that student group as the context.'
    return "No previous context is available."

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

Student Records:
{chr(10).join(student_blocks)}

Write a polished, helpful response with this structure:

1. Start immediately with a direct answer in a normal sentence. Do not use the heading "Direct Answer".
2. In the next paragraph, directly describe the matching student profile(s). Do not use the heading "What the Records Show".
3. If the user asks about SAT, ACT, AMC, AP courses, academics, boards, grades, extracurricular activities, summer programs, academic programs, projects, leadership, financial aid, or scholarships, answer using the "Requested Follow-up Details" provided in each student record.
4. If teacher names are explicitly mentioned in the Advice text, include a plain-text section titled "Teacher and Mentor Support".
5. Include a plain-text section titled "Patterns and Takeaways" when there is enough relevant information.
6. Include a plain-text section titled "Practical Guidance".
7. End with one helpful, self-contained follow-up question the user could ask next. The follow-up question should include the university, school, or student context where possible, not vague phrases like "this student".

Important rules:
- Do not invent universities, scores, schools, activities, outcomes, teachers, EE supervisors, or other facts.
- Do not use outside knowledge.
- You may synthesize patterns from the provided records.
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
@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    user_query = clean_user_query(req.message)
    session_id = get_session_key(req)
    intent = classify_intent(user_query)

    if intent == "advisory":
        return {
            "intent": "advisory",
            "assistant_answer": "Advisory flow unchanged."
        }

    all_columns = sheet.row_values(1)
    normalized_columns = [normalize(col).replace(" ", "_") for col in all_columns]

    prompt = f"""
Convert the user query into JSON.
Allowed keys:
{', '.join(normalized_columns)}

User query:
"{user_query}"
"""

    raw = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    filters = json.loads(match.group()) if match else {}

    # Handle queries like "Greenwood High School admitted into US colleges".
    # This means: country = USA and at least one final admitted university.
    # It should not be treated as a university named "US colleges".
    filters = sanitize_filters_for_country_college_query(user_query, filters)

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

    if (
        "admitted_university" not in mapped_filters
        and re.search(r"\b(admit|admitted|accepted|got into|get into)\b", user_query.lower())
    ):
        admit_target = extract_admit_target_from_query(user_query)
        if admit_target:
            filters["admitted_university"] = admit_target

    filters = sanitize_filters_for_country_college_query(user_query, filters)

    if (
        intent == "analytics"
        and "school_name" not in mapped_filters
        and re.search(r"\bschool\b", user_query.lower())
    ):
        match = re.search(r"from\s+(.+?school)", user_query, re.IGNORECASE)
        if match:
            filters["school_name"] = match.group(1).strip()

    # If this is a follow-up question, inherit the previous admit/school context
    # unless the current question explicitly provides a new one.
    filters = apply_memory_to_filters(user_query, filters, session_id)

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    # Save useful context for the next turn, e.g. Cornell admits -> SAT scores / financial aid.
    update_memory(session_id, user_query, filters, students)

    base_response = handle_analytics_response(user_query, students)
    return generate_nlg_response(user_query, students, base_response, session_id=session_id)
