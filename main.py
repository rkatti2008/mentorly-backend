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
        "can you advise", "what should i do", "guidance", "counsel",
        "advice", "i am a student", "i want to apply", "what are my chances"
    ]):
        return "advisory"

    if any(k in q for k in ["how many", "count", "number of", "statistics"]):
        return "analytics"

    return "hybrid"

# -------------------------------
# Helpers
# -------------------------------
def fuzzy_match(text, pattern, threshold=0.7):
    if not text or not pattern:
        return False
    text = str(text).lower()
    pattern = str(pattern).lower()
    return pattern in text or SequenceMatcher(None, text, pattern).ratio() >= threshold

def get_column_value(row: dict, key: str):
    key_norm = key.lower().strip()
    for col in row:
        if col.lower().strip() == key_norm:
            return row[col]
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

            if key.lower() == "intended_major":
                majors = [m.strip() for m in val.split(",")]
                cell = get_column_value(r, "Intended Major")
                if not any(fuzzy_match(cell, m) for m in majors):
                    include = False
                    break

            elif key.lower() == "countries applied to":
                cell = r.get("Countries Applied To", "")
                countries = [c.strip().lower() for c in cell.split(",") if c.strip()]
                vals = val if isinstance(val, list) else [val]
                if not any(v.lower() in countries for v in vals):
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
# 🚨 Phase 6.2.4 — Unsafe Analytics Guard
# -------------------------------
def mentions_unsupported_dimension(query: str) -> bool:
    q = query.lower()

    # crude but safe detection
    school_words = ["school", "high", "dps", "greenwood"]
    uni_words = ["mit", "harvard", "stanford", "cornell", "princeton"]

    return any(w in q for w in school_words + uni_words)

def safe_blocked_analytics_response():
    return {
        "intent": "analytics",
        "assistant_answer": (
            "I can’t answer this reliably yet because the current data filters "
            "don’t support school- or university-specific counts. "
            "You can ask about broader trends (by country, board, or major), "
            "or we can extend the system to support this."
        )
    }

# -------------------------------
# Analytics Narrator
# -------------------------------
def handle_analytics_response(user_query, students):
    count = len(students)

    prompt = f"""
Answer the user's question clearly and factually.

Question:
"{user_query}"

Answer rules:
- Start with the number
- One short sentence
- No speculation
"""

    resp = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=80
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
        return handle_advisory(req.message)

    # 🚨 Block unsafe analytics
    if intent == "analytics" and mentions_unsupported_dimension(req.message):
        return safe_blocked_analytics_response()

    prompt = f"""
Convert the user query into JSON filters.

Allowed keys:
intended_major,
countries applied to

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

    return handle_hybrid(req.message, {})
