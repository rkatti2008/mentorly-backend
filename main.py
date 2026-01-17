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
# ✅ TRUE FIX: ROW-WIDE SEARCH
# -------------------------------
def row_contains_value(row: dict, query: str) -> bool:
    for cell in row.values():
        if fuzzy_match(str(cell), query):
            return True
    return False

def row_contains_university(row: dict, query: str) -> bool:
    q_norm = normalize_university(query)
    for cell in row.values():
        cell_norm = normalize_university(str(cell))
        if fuzzy_match(cell_norm, q_norm):
            return True
    return False

# -------------------------------
# Core Filter Engine (FINAL)
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
