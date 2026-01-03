from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json
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
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SHEET_ID = os.environ.get("SHEET_ID")

if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
if not SHEET_ID:
    raise RuntimeError("SHEET_ID not set")

creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
)

client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")

client_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Phase 6 — Intent Classification
# -------------------------------
def classify_intent(user_query: str) -> str:
    q = user_query.lower()

    advisory_keywords = [
        "can you advise",
        "what should i do",
        "guidance",
        "counsel",
        "advice",
        "i am a student",
        "i want to apply",
        "help me choose",
        "what are my chances"
    ]

    analytics_keywords = [
        "how many",
        "count",
        "number of",
        "statistics",
        "analytics"
    ]

    if any(k in q for k in advisory_keywords):
        return "advisory"

    if any(k in q for k in analytics_keywords):
        return "analytics"

    return "hybrid"

# -------------------------------
# Phase 6.1 — Pure Counselor Advice
# -------------------------------
def handle_advisory(user_query: str) -> dict:
    prompt = f"""
You are an experienced international college counselor.

The student is asking for personal guidance.
Do NOT mention databases, analytics, counts, or other students.
Do NOT fabricate statistics.

Respond like a real counselor:
- Calm
- Structured
- Encouraging
- Action-oriented

Student query:
"{user_query}"

Provide:
1. Short reassurance
2. Key considerations
3. Next concrete steps
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=400
    )

    return {
        "intent": "advisory",
        "assistant_answer": response.choices[0].message.content.strip()
    }

# -------------------------------
# Helpers
# -------------------------------
def fuzzy_match(text, pattern, threshold=0.7):
    if not text or not pattern:
        return False

    text = str(text).lower()
    pattern = str(pattern).lower()

    if pattern in text:
        return True

    return SequenceMatcher(None, text, pattern).ratio() >= threshold


def get_column_value(row: dict, key: str):
    key_norm = key.lower().strip()
    for col in row:
        if col.lower().strip() == key_norm:
            return row[col]
    return None

# -------------------------------
# Core filter engine
# -------------------------------
def filter_students(records, query_params):
    filtered = []

    for r in records:
        include = True

        for key, val in query_params.items():
            if val is None:
                continue

            if key.lower() == "intended_major":
                majors = [m.strip() for m in val.split(",")]
                cell = get_column_value(r, "Intended Major")
                if not any(fuzzy_match(cell, m) for m in majors):
                    include = False
                    break

            elif key.lower() == "countries applied to":
                cell = r.get("Countries Applied To", "")
                if not fuzzy_match(cell, val):
                    include = False
                    break

        if include:
            filtered.append(r)

    return filtered

# -------------------------------
# Analytics
# -------------------------------
def compute_analytics(students: list) -> dict:
    if not students:
        return {}

    return {
        "countries_applied": Counter(
            s.get("Countries Applied To", "").strip().lower()
            for s in students if s.get("Countries Applied To")
        ),
        "intended_majors": Counter(
            s.get("Intended Major", "").strip().lower()
            for s in students if s.get("Intended Major")
        ),
    }

# -------------------------------
# Phase 6.2.2 — Analytics Narration
# -------------------------------
def generate_analytics_answer(user_query: str, count: int, filters: dict) -> str:
    q = user_query.lower()

    if count == 0:
        return (
            "I couldn’t find any students matching this exact combination. "
            "You may want to broaden the criteria slightly (for example, remove one filter) "
            "to see broader patterns."
        )

    parts = []

    if "ib" in q:
        parts.append("IB students")
    else:
        parts.append("students")

    if "usa" in q or "united states" in q:
        parts.append("who applied to the USA")

    sentence = " ".join(parts)

    return f"Here’s what I found: **{count} {sentence}**."

# -------------------------------
# Phase 6.2.1 — Hybrid Signals
# -------------------------------
def summarize_signals(analytics: dict) -> dict:
    if not analytics:
        return {}

    return {
        "popular_countries": list(analytics.get("countries_applied", {}).keys()),
        "common_majors": list(analytics.get("intended_majors", {}).keys())
    }


def handle_hybrid(user_query: str, signals: dict) -> dict:
    prompt = f"""
You are a senior international college counselor.

The student wants advice, informed by general trends.
Do NOT mention counts, percentages, or databases.
Use patterns only as soft context.

Student query:
"{user_query}"

Contextual signals:
{json.dumps(signals, indent=2)}

Respond with:
1. Understanding of the student's background
2. Strategic advice
3. Clear next steps
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=450
    )

    return {
        "intent": "hybrid",
        "assistant_answer": response.choices[0].message.content.strip()
    }

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):

    intent = classify_intent(req.message)

    if intent == "advisory":
        return handle_advisory(req.message)

    # ---------- Filter extraction ----------
    prompt = f"""
Convert the user query into JSON filters.

Allowed keys:
intended_major,
countries applied to

User query:
"{req.message}"
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content
    filters = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])

    records = sheet.get_all_records()
    students = filter_students(records, filters)
    analytics = compute_analytics(students)

    if intent == "analytics":
        return {
            "intent": "analytics",
            "assistant_answer": generate_analytics_answer(
                req.message, len(students), filters
            ),
            "interpreted_filters": filters,
            "count": len(students),
            "analytics": analytics
        }

    signals = summarize_signals(analytics)
    return handle_hybrid(req.message, signals)
