from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json
import re

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
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SHEET_ID = os.environ.get("SHEET_ID")

creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
)

client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

client_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Intent Classification
# -------------------------------
def classify_intent(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["how many", "count", "number of", "statistics"]):
        return "analytics"
    if any(k in q for k in ["advice", "guidance", "counsel", "what should i do"]):
        return "advisory"
    return "hybrid"

# -------------------------------
# Text Helpers
# -------------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\"\'“”.,()\-]', '', text)
    return text.strip()

def fuzzy_match(a: str, b: str, threshold=0.72) -> bool:
    a = normalize(a)
    b = normalize(b)
    if not a or not b:
        return False
    return (
        b in a
        or a in b
        or SequenceMatcher(None, a, b).ratio() >= threshold
    )

# -------------------------------
# Clean user query (quoted input fix)
# -------------------------------
def clean_user_query(q: str) -> str:
    q = q.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        q = q[1:-1]
    return q.strip()

# -------------------------------
# University Normalization
# -------------------------------
UNIVERSITY_ALIASES = {
    "ucsd": [
        "university of california san diego",
        "uc san diego",
        "university of california san diego ucsd"
    ],
    "mit": ["massachusetts institute of technology"],
    "cornell": ["cornell university"],
    "uc berkeley": ["university of california berkeley", "uc berkeley"]
}

def normalize_university(val: str) -> str:
    val = normalize(val)
    for canon, aliases in UNIVERSITY_ALIASES.items():
        if val == canon or val in aliases:
            return canon
    return val

# -------------------------------
# ✅ FINAL ADMISSION COLUMN LOGIC (CORRECT)
# -------------------------------
ADMIT_INCLUDE_HINTS = [
    "admit",
    "admitted",
    "final",
    "decision",
    "result"
]

ADMIT_EXCLUDE_HINTS = [
    "applied",
    "application",
    "preference",
    "choice",
    "list"
]

def is_admit_column(col_name: str) -> bool:
    col = normalize(col_name)

    # ❌ Never count applied / preference columns
    if any(bad in col for bad in ADMIT_EXCLUDE_HINTS):
        return False

    # ✅ Only explicit admission outcome columns
    return any(good in col for good in ADMIT_INCLUDE_HINTS)

# -------------------------------
# Row Matching
# -------------------------------
def row_contains_value(row: dict, query: str) -> bool:
    for cell in row.values():
        if fuzzy_match(str(cell), query):
            return True
    return False

def row_contains_university(row: dict, query: str) -> bool:
    q_norm = normalize_university(query)

    for col_name, cell in row.items():
        if not is_admit_column(col_name):
            continue  # 🔒 prevents "applied to" false positives

        cell_text = normalize(str(cell))

        if q_norm in cell_text:
            return True

        if fuzzy_match(cell_text, q_norm):
            return True

    return False

# -------------------------------
# Core Filter Engine
# -------------------------------
def filter_students(records, filters):
    result = []

    for row in records:
        ok = True

        if "school_name" in filters:
            if not row_contains_value(row, filters["school_name"]):
                ok = False

        if ok and "admitted_university" in filters:
            if not row_contains_university(row, filters["admitted_university"]):
                ok = False

        if ok:
            result.append(row)

    return result

# -------------------------------
# Analytics Response
# -------------------------------
def handle_analytics_response(query, students):
    prompt = f"""
User question:
"{query}"

Exact count:
{len(students)}

Rules:
- Start with the number
- One sentence only
"""
    resp = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60
    )

    return {
        "intent": "analytics",
        "assistant_answer": resp.choices[0].message.content.strip()
    }

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    user_query = clean_user_query(req.message)
    intent = classify_intent(user_query)

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
    resp = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    filters = json.loads(match.group()) if match else {}

    for k, v in filters.items():
        if isinstance(v, str):
            filters[k] = normalize(v)

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    return handle_analytics_response(user_query, students)
