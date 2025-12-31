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
        include = True

        # IB filter
        if query_params.get("ib_min_12") is not None or query_params.get("ib_max_12") is not None:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            if not passes_numeric_filter(
                r.get("12th grade overall score"),
                query_params.get("ib_min_12"),
                query_params.get("ib_max_12"),
            ):
                continue

        # SAT filter
        if sat_used:
            if not passes_numeric_filter(
                r.get("SAT Total score"),
                query_params.get("SAT Total score_min"),
                query_params.get("SAT Total score_max"),
            ):
                continue

        # ACT filter
        if act_used:
            if not passes_numeric_filter(
                r.get("ACT Score"),
                query_params.get("ACT Score_min"),
                query_params.get("ACT Score_max"),
            ):
                continue

        # Other filters
        for key, val in query_params.items():
            if key in [
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
# Phase 3.4 — Filter normalization
# -------------------------------
def normalize_filters(filters: dict) -> dict:
    normalized = {}

    COUNTRY_SYNONYMS = {
        "america": "usa",
        "united states": "usa",
        "united states of america": "usa",
        "us": "usa",
        "u.s.": "usa"
    }

    MAJOR_SYNONYMS = {
        "engineering": (
            "mechanical engineering, electrical engineering, "
            "civil engineering, chemical engineering, "
            "computer engineering, aerospace engineering"
        ),
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
# Phase 3.4.3 — Repair LLM mistakes
# -------------------------------
def repair_llm_filters(filters: dict, user_query: str) -> dict:
    repaired = dict(filters)

    COUNTRY_WORDS = {"america", "usa", "us", "united states", "uk", "canada", "india"}

    # Prevent IB being treated as a university
    if "admitted univs" in repaired:
        val = repaired["admitted univs"]
        if isinstance(val, str) and val.lower() == "ib":
            repaired.pop("admitted univs")

    if "admitted univs" in repaired:
        val = repaired["admitted univs"]
        if val is not None:
            if isinstance(val, str):
                val = [val]
            val_lower = [v.lower() for v in val if v]
            if any(v in COUNTRY_WORDS for v in val_lower):
                repaired.pop("admitted univs")
                repaired["countries applied to"] = val_lower[0]
            else:
                repaired["admitted univs"] = val

    if "ib" in user_query.lower():
        repaired.setdefault("ib_min_12", 24)
        repaired.setdefault("ib_max_12", 45)

    return repaired

# -------------------------------
# Phase 5.3 — Analytics
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
# Phase 5.7 — Counselor Explanation
# -------------------------------
def generate_counselor_explanation(filters, students, analytics):
    prompt = f"""
You are an experienced college counselor.

Explain the results clearly.
If results are zero, explain likely semantic reasons and suggest how to broaden the query.
Engineering includes Mechanical, Electrical, Civil, Chemical, etc.
Do NOT invent student data.

Filters used:
{json.dumps(filters, indent=2)}

Result count:
{len(students)}

Analytics:
{json.dumps(analytics, indent=2)}

Write 3–5 sentences.
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

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

    assistant_answer = generate_counselor_explanation(
        filters=filters,
        students=students,
        analytics=analytics
    )

    return {
        "interpreted_filters": filters,
        "count": len(students),
        "assistant_answer": assistant_answer,
        "analytics": analytics,
        "students": students
    }
