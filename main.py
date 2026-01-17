from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json
from collections import Counter
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

    if any(k in q for k in [
        "can you advise", "what should i do", "guidance",
        "counsel", "advice", "i am a student",
        "i want to apply", "what are my chances"
    ]):
        return "advisory"

    if any(k in q for k in ["how many", "count", "number of", "statistics"]):
        return "analytics"

    return "hybrid"

# -------------------------------
# Helpers
# -------------------------------
def normalize_text(val: str) -> str:
    if not val:
        return ""
    val = val.lower()
    val = re.sub(r'[\"\'“”.,]', '', val)
    return val.strip()

def fuzzy_match(text, pattern, threshold=0.7):
    if not text or not pattern:
        return False
    text = normalize_text(text)
    pattern = normalize_text(pattern)
    return pattern in text or SequenceMatcher(None, text, pattern).ratio() >= threshold

def get_column_value(row: dict, key: str):
    key_norm = key.lower().strip()
    for col in row:
        if col.lower().strip() == key_norm:
            return row[col]
    return ""

# -------------------------------
# Admitted University (Phase 6.3.1)
# -------------------------------
ADMIT_COLUMNS = [
    "Final University",
    "University Admitted",
    "Admitted To",
    "College",
    "University"
]

UNIVERSITY_ALIASES = {
    "mit": ["massachusetts institute of technology"],
    "ucsd": ["university of california san diego", "uc san diego"],
    "cornell": ["cornell university"],
    "uc berkeley": ["university of california berkeley", "uc berkeley"]
}

def detect_admit_column(row: dict) -> str | None:
    for col in row:
        if col.strip() in ADMIT_COLUMNS:
            return col
    return None

def normalize_university(name: str) -> str:
    name = normalize_text(name)
    for canonical, aliases in UNIVERSITY_ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name

# -------------------------------
# ✅ School Column Fix (Phase 6.3.2)
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
# Phase 6.3 — Core Filter Engine
# -------------------------------
def filter_students(records, query_params):
    filtered = []

    for r in records:
        include = True

        for key, val in query_params.items():
            if not val:
                continue

            # Intended Major
            if key == "intended_major":
                cell = get_column_value(r, "Intended Major")
                if not fuzzy_match(cell, val):
                    include = False
                    break

            # Country
            elif key == "countries_applied_to":
                cell = r.get("Countries Applied To", "")
                countries = [c.strip().lower() for c in cell.split(",")]
                if val.lower() not in countries:
                    include = False
                    break

            # ✅ FIXED: School
            elif key == "school_name":
                school_col = detect_school_column(r)
                if not school_col:
                    include = False
                    break

                cell = r.get(school_col, "")
                if not fuzzy_match(cell, val):
                    include = False
                    break

            # Admitted University
            elif key == "admitted_university":
                admit_col = detect_admit_column(r)
                if not admit_col:
                    include = False
                    break

                cell = r.get(admit_col, "")
                if normalize_university(cell) != normalize_university(val):
                    include = False
                    break

        if include:
            filtered.append(r)

    return filtered

# -------------------------------
# Board Filter
# -------------------------------
def apply_board_filter(students, query):
    q = query.lower()
    if "ib" in q:
        return [s for s in students if "ib" in s.get("12th Board", "").lower()]
    if "cbse" in q:
        return [s for s in students if "cbse" in s.get("12th Board", "").lower()]
    return students

# -------------------------------
# Analytics Narrator (SAFE)
# -------------------------------
def handle_analytics_response(user_query, students):
    count = len(students)

    prompt = f"""
User question:
"{user_query}"

Exact count:
{count}

Rules:
- Start with the number
- One sentence only
- No assumptions
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
# Advisory
# -------------------------------
def handle_advisory(query):
    prompt = f"""
You are a senior international college counselor.

Student query:
"{query}"

Provide thoughtful, personalized advice.
Do NOT mention other students or statistics.
"""

    resp = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=400
    )

    return {
        "intent": "advisory",
        "assistant_answer": resp.choices[0].message.content.strip()
    }

# -------------------------------
# Hybrid
# -------------------------------
def handle_hybrid(query):
    return handle_advisory(query)

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):

    intent = classify_intent(req.message)

    if intent == "advisory":
        return handle_advisory(req.message)

    prompt = f"""
Convert the user query into JSON.

Allowed keys:
school_name,
admitted_university,
intended_major,
countries_applied_to

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
    students = apply_board_filter(students, req.message)

    if intent == "analytics":
        return handle_analytics_response(req.message, students)

    return handle_hybrid(req.message)
