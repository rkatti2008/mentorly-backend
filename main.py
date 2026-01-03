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
# Phase 6.1 — Intent Classification
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

    if any(k in q for k in advisory_keywords):
        return "advisory"

    return "analytics"

# -------------------------------
# Phase 6.1 — Counselor-only advice
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
def passes_numeric_filter(value_raw, min_val=None, max_val=None):
    try:
        value = int(value_raw)
    except (TypeError, ValueError):
        return False

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


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
# Core filter engine (LOCKED)
# -------------------------------
def filter_students(records, query_params):
    filtered = []

    sat_vals = [
        query_params.get("SAT Total score_min"),
        query_params.get("SAT Total score_max"),
    ]
    act_vals = [
        query_params.get("ACT Score_min"),
        query_params.get("ACT Score_max"),
    ]

    sat_used = any(v is not None for v in sat_vals)
    act_used = any(v is not None for v in act_vals)

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:

        if query_params.get("ib_board_only"):
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

        if query_params.get("cbse_board_only"):
            if r.get("12th Board", "").strip().upper() != "CBSE":
                continue

        if query_params.get("ib_min_12") is not None or query_params.get("ib_max_12") is not None:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            if not passes_numeric_filter(
                r.get("12th grade overall score"),
                query_params.get("ib_min_12"),
                query_params.get("ib_max_12"),
            ):
                continue

        if sat_used:
            if not passes_numeric_filter(
                r.get("SAT Total score"),
                query_params.get("SAT Total score_min"),
                query_params.get("SAT Total score_max"),
            ):
                continue

        if act_used:
            if not passes_numeric_filter(
                r.get("ACT Score"),
                query_params.get("ACT Score_min"),
                query_params.get("ACT Score_max"),
            ):
                continue

        include = True

        for key, val in query_params.items():
            if key in [
                "ib_board_only", "cbse_board_only",
                "ib_min_12", "ib_max_12",
                "SAT Total score_min", "SAT Total score_max",
                "ACT Score_min", "ACT Score_max"
            ]:
                continue

            if val is None:
                continue

            if key.lower() == "intended_major":
                majors = [m.strip() for m in val.split(",")]
                cell = get_column_value(r, "Intended Major")
                if not any(fuzzy_match(cell, m) for m in majors):
                    include = False
                    break

            elif key.lower() == "city":
                if r.get("City of Graduation", "").lower() != val.lower():
                    include = False
                    break

            elif key.lower() == "admitted univs":
                cell = r.get("Accepted Univ", "")
                if isinstance(val, list):
                    if not any(fuzzy_match(cell, v) for v in val):
                        include = False
                        break
                else:
                    if not fuzzy_match(cell, val):
                        include = False
                        break

            elif key.lower() == "countries applied to":
                cell = r.get("Countries Applied To", "")
                if not fuzzy_match(cell, val):
                    include = False
                    break

            else:
                cell = get_column_value(r, key)
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
        "admitted_universities": Counter(
            s.get("Accepted Univ", "").strip().lower()
            for s in students if s.get("Accepted Univ")
        ),
    }

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):

    intent = classify_intent(req.message)

    if intent == "advisory":
        return handle_advisory(req.message)

    # ---------- Phase 5 analytics pipeline ----------
    prompt = f"""
Convert the user query into JSON filters.

Allowed keys:
ib_min_12, ib_max_12,
"SAT Total score_min", "SAT Total score_max",
"ACT Score_min", "ACT Score_max",
intended_major,
admitted univs,
countries applied to,
city

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

    return {
        "intent": "analytics",
        "interpreted_filters": filters,
        "count": len(students),
        "analytics": analytics,
        "students": students
    }
