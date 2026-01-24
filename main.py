from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
import os, json, re
from collections import Counter

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
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Intent + Mode Classification
# -------------------------------
def classify_intent(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["how many", "count", "number of"]):
        return "analytics"
    if any(k in q for k in ["advice", "guidance", "counsel"]):
        return "advisory"
    return "hybrid"

def classify_mode(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["why", "how do", "is it possible", "chances", "realistic"]):
        return "hybrid"
    return "analytics"

def is_exclusive_admit(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["only", "sole", "final choice", "chose", "exclusively"])

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
        if not is_school_column(col):
            continue
        if target in normalize(cell):
            return True
    return False

# -------------------------------
# Admit Column Logic
# -------------------------------
ADMIT_INCLUDE_HINTS = ["final", "admit", "admitted", "decision", "result"]
ADMIT_EXCLUDE_HINTS = ["applied", "application", "preference", "choice", "list"]

def is_admit_column(col_name: str) -> bool:
    col = normalize(col_name)
    if any(bad in col for bad in ADMIT_EXCLUDE_HINTS):
        return False
    return any(good in col for good in ADMIT_INCLUDE_HINTS)

def extract_universities(cell: str):
    return [
        normalize_university(p.strip())
        for p in re.split(r"[;,/|]", normalize(cell))
        if p.strip()
    ]

# Inclusive admit (Phase-6 compatible)
def row_has_admit(row: dict, university: str) -> bool:
    target = normalize_university(university)
    for col, cell in row.items():
        if not is_admit_column(col):
            continue
        if target in extract_universities(cell):
            return True
    return False

# Exclusive admit (Phase-7 strict)
def row_has_exclusive_final_admit(row: dict, university: str) -> bool:
    target = normalize_university(university)
    for col, cell in row.items():
        if not is_admit_column(col):
            continue
        universities = extract_universities(cell)
        if len(universities) == 1 and universities[0] == target:
            return True
    return False

# -------------------------------
# Core Filter Engine
# -------------------------------
def filter_students(records, filters, exclusive=False):
    result = []
    for row in records:
        ok = True

        if "school_name" in filters:
            if not row_has_school(row, filters["school_name"]):
                ok = False

        if ok and "admitted_university" in filters:
            if exclusive:
                if not row_has_exclusive_final_admit(row, filters["admitted_university"]):
                    ok = False
            else:
                if not row_has_admit(row, filters["admitted_university"]):
                    ok = False

        if ok:
            result.append(row)

    return result

# -------------------------------
# Phase 7: Explainable Summaries
# -------------------------------
def summarize_patterns(students):
    boards = Counter()
    schools = Counter()

    for s in students:
        boards[s.get("12th Board", "Unknown")] += 1
        schools[s.get("School", "Unknown")] += 1

    return {
        "top_boards": [b for b, _ in boards.most_common(2)],
        "top_schools": [s for s, _ in schools.most_common(2)]
    }

# -------------------------------
# Responses
# -------------------------------
def analytics_response(students):
    if len(students) == 0:
        return "0 students match the criteria."
    return f"{len(students)} students match the criteria."

def hybrid_response(summary):
    prompt = f"""
Facts (do not invent anything):
- Boards seen: {summary['top_boards']}
- Schools seen: {summary['top_schools']}

Rules:
- No numbers
- No guarantees
- Use cautious language like "tend to", "typically"
"""
    return client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    user_query = clean_user_query(req.message)
    intent = classify_intent(user_query)
    mode = classify_mode(user_query)
    exclusive = is_exclusive_admit(user_query)

    if intent == "advisory":
        return {
            "intent": "advisory",
            "assistant_answer": "Advisory flow unchanged."
        }

    prompt = f"""
Convert the user query into JSON.
Allowed keys:
school_name,
admitted_university

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

    records = sheet.get_all_records()
    students = filter_students(records, filters, exclusive=exclusive)

    if mode == "hybrid":
        answer = hybrid_response(summarize_patterns(students))
    else:
        answer = analytics_response(students)

    return {
        "intent": intent,
        "assistant_answer": answer
    }
