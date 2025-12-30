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

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")

client_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str

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

    # ✅ Only raise error if numeric filters for both SAT and ACT exist
    sat_used = "SAT Total score_min" in query_params or "SAT Total score_max" in query_params
    act_used = "ACT Score_min" in query_params or "ACT Score_max" in query_params

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        if "ib_min_12" in query_params or "ib_max_12" in query_params:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            min_ib = query_params.get("ib_min_12")
            max_ib = query_params.get("ib_max_12")

            if not passes_numeric_filter(
                r.get("12th grade overall score"),
                min_ib,
                max_ib,
            ):
                continue

        if "SAT Total score_min" in query_params or "SAT Total score_max" in query_params:
            if not passes_numeric_filter(
                r.get("SAT Total score"),
                query_params.get("SAT Total score_min"),
                query_params.get("SAT Total score_max"),
            ):
                continue

        if "ACT Score_min" in query_params or "ACT Score_max" in query_params:
            if not passes_numeric_filter(
                r.get("ACT Score"),
                query_params.get("ACT Score_min"),
                query_params.get("ACT Score_max"),
            ):
                continue

        for key, val in query_params.items():
            if key in [
                "ib_min_12", "ib_max_12",
                "SAT Total score_min", "SAT Total score_max",
                "ACT Score_min", "ACT Score_max"
            ]:
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

            else:
                cell = get_column_value(r, key)
                if not fuzzy_match(cell, val):
                    include = False
                    break

        if include:
            filtered.append(r)

    return filtered

# -------------------------------
# Phase 3.4 — Filter normalization
# -------------------------------
def normalize_filters(filters: dict) -> dict:
    normalized = {}

    COUNTRY_SYNONYMS = {
        "america": "usa",
        "amrika": "usa",
        "united states": "usa",
        "united states of america": "usa",
        "us": "usa",
        "u.s.": "usa"
    }

    MAJOR_SYNONYMS = {
        "cs": "computer science",
        "comp sci": "computer science",
        "cse": "computer science"
    }

    for key, val in filters.items():
        if isinstance(val, str):
            v = val.strip().lower()
            if key == "countries applied to":
                v = COUNTRY_SYNONYMS.get(v, v)
            if key == "intended_major":
                v = MAJOR_SYNONYMS.get(v, v)
            normalized[key] = v
        else:
            normalized[key] = val

    return normalized

# -------------------------------
# Phase 3.4.3 — Repair LLM mistakes (IB and numeric guards)
# -------------------------------
def repair_llm_filters(filters: dict, user_query: str) -> dict:
    repaired = dict(filters)

    COUNTRY_WORDS = {"america", "usa", "us", "united states", "uk", "canada", "india"}

    if "admitted univs" in repaired:
        val = str(repaired["admitted univs"]).lower().strip()
        if val in COUNTRY_WORDS:
            repaired.pop("admitted univs")
            repaired["countries applied to"] = val

    query_lower = user_query.lower()

    # Correct IB diploma defaults
    if "ib" in query_lower:
        if "ib_min_12" not in repaired:
            repaired["ib_min_12"] = 24
        if "ib_max_12" not in repaired:
            repaired["ib_max_12"] = 45

    # Ensure IB numeric bounds are reasonable
    if repaired.get("ib_max_12") is not None and repaired["ib_max_12"] < 24:
        repaired["ib_max_12"] = 45
    if repaired.get("ib_min_12") is not None and repaired["ib_min_12"] < 24:
        repaired["ib_min_12"] = 24

    return repaired

# -------------------------------
# Phase 5.3.2 — Analytics Helpers
# -------------------------------
def compute_analytics(students: list) -> dict:
    if not students:
        return {}

    analytics = {}

    analytics["countries_applied"] = Counter(
        s.get("Countries Applied To", "").strip().lower()
        for s in students if s.get("Countries Applied To")
    )

    analytics["intended_majors"] = Counter(
        s.get("Intended Major", "").strip().lower()
        for s in students if s.get("Intended Major")
    )

    analytics["admitted_universities"] = Counter(
        s.get("Accepted Univ", "").strip().lower()
        for s in students if s.get("Accepted Univ")
    )

    ib_scores = [
        int(s.get("12th grade overall score"))
        for s in students
        if s.get("12th Board", "").strip().upper() == "IBDP"
        and str(s.get("12th grade overall score")).isdigit()
    ]

    if ib_scores:
        analytics["ib_score_range"] = {
            "min": min(ib_scores),
            "max": max(ib_scores),
            "average": round(sum(ib_scores) / len(ib_scores), 2)
        }

    return analytics

# -------------------------------
# POST /nl_query
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):

    prompt = f"""
You are a strict JSON generator.
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

    filters = repair_llm_filters(filters, req.message)
    filters = normalize_filters(filters)

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    analytics = compute_analytics(students)

    if not students:
        assistant_answer = "No matching student records were found for your query."
    else:
        assistant_answer = f"{len(students)} matching student records were found."

    return {
        "interpreted_filters": filters,
        "count": len(students),
        "assistant_answer": assistant_answer,
        "analytics": analytics,
        "students": students
    }
