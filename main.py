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
def classify_intent(user_query: str) -> str:
    q = user_query.lower()
    if any(k in q for k in ["how many", "count", "number of", "statistics"]):
        return "analytics"
    if any(k in q for k in [
        "can you advise", "guidance", "counsel", "advice",
        "what should i do", "what are my chances"
    ]):
        return "advisory"
    return "hybrid"

# -------------------------------
# Helpers
# -------------------------------
def normalize_text(val: str) -> str:
    if not val:
        return ""
    val = val.lower()
    val = re.sub(r'[\"\'“”.,()]', '', val)
    return val.strip()

def fuzzy_match(text, pattern, threshold=0.7):
    if not text or not pattern:
        return False
    text = normalize_text(text)
    pattern = normalize_text(pattern)
    return (
        pattern in text
        or text in pattern
        or SequenceMatcher(None, text, pattern).ratio() >= threshold
    )

# -------------------------------
# University Logic
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

def normalize_university(name: str) -> str:
    name = normalize_text(name)
    for canonical, aliases in UNIVERSITY_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name

def university_matches(cell_value: str, query_value: str) -> bool:
    cell_norm = normalize_university(cell_value)
    query_norm = normalize_university(query_value)

    if cell_norm == query_norm:
        return True

    return fuzzy_match(cell_norm, query_norm)

# -------------------------------
# ✅ FIXED: Robust Admit Column Detection
# -------------------------------
def detect_admit_column(row: dict) -> str | None:
    """
    Robustly detect final admitted university column.
    Handles real-world Google Sheet headers.
    """

    priority_keywords = [
        "final",
        "attend",
        "admit",
        "committed"
    ]

    fallback_keywords = [
        "university",
        "college"
    ]

    cols = list(row.keys())

    # Strong signals first
    for col in cols:
        col_norm = col.lower()
        if any(k in col_norm for k in priority_keywords):
            return col

    # Fallback
    for col in cols:
        col_norm = col.lower()
        if any(k in col_norm for k in fallback_keywords):
            return col

    return None

# -------------------------------
# School Logic
# -------------------------------
SCHOOL_COLUMNS = [
    "School",
    "12th School",
    "High School",
    "School Name",
    "Secondary School"
]

def detect_school_column(row: dict) -> str | None:
    for col in row:
        if col.strip().lower() in [c.lower() for c in SCHOOL_COLUMNS]:
            return col
    return None

# -------------------------------
# Core Filter Engine
# -------------------------------
def filter_students(records, query_params):
    filtered = []

    for r in records:
        include = True

        for key, val in query_params.items():
            if not val:
                continue

            if key == "school_name":
                col = detect_school_column(r)
                if not col or not fuzzy_match(r.get(col, ""), val):
                    include = False
                    break

            elif key == "admitted_university":
                col = detect_admit_column(r)
                if not col or not university_matches(r.get(col, ""), val):
                    include = False
                    break

        if include:
            filtered.append(r)

    return filtered

# -------------------------------
# Analytics Response
# -------------------------------
def handle_analytics_response(user_query, students):
    prompt = f"""
User question:
"{user_query}"

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
    intent = classify_intent(req.message)

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
"{req.message}"
"""
    resp = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content
    filters = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    return handle_analytics_response(req.message, students)
