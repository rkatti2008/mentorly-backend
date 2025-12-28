from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json
import re
from openai import OpenAI

app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
SHEET_ID = os.environ.get("SHEET_ID")

if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("SERVICE_ACCOUNT_JSON not set")

creds = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# -------------------------------
# OpenAI Client
# -------------------------------
llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------
# Request models
# -------------------------------
class NLQueryRequest(BaseModel):
    query: str

# -------------------------------
# Helper functions
# -------------------------------
def passes_numeric_filter(value_raw, min_val=None, max_val=None):
    try:
        value = int(value_raw)
    except (TypeError, ValueError):
        return False if (min_val is not None or max_val is not None) else True

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


def fuzzy_match(text, patterns, threshold=0.7, multiple=False):
    if patterns is None:
        return True
    if text is None:
        return False

    text = str(text).lower()
    patterns = [p.strip().lower() for p in str(patterns).split(",")] if multiple else [str(patterns).lower()]

    for pattern in patterns:
        if pattern in text:
            return True
        if SequenceMatcher(None, text, pattern).ratio() >= threshold:
            return True

    return False


def extract_json_from_llm(text: str) -> dict:
    """
    Extract first valid JSON object from LLM output
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    return json.loads(match.group())


# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    filtered = []
    query_params = dict(request.query_params)

    sat_keys = ["SAT Total score", "SAT Math", "SAT English"]
    act_keys = ["ACT Score"]

    sat_filter = any(f"{k}_min" in query_params or f"{k}_max" in query_params for k in sat_keys)
    act_filter = any(f"{k}_min" in query_params or f"{k}_max" in query_params for k in act_keys)

    if sat_filter and act_filter:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        for key, value in query_params.items():
            if key.endswith("_min") or key.endswith("_max"):
                col = key.rsplit("_", 1)[0]
                min_val = int(query_params.get(f"{col}_min")) if f"{col}_min" in query_params else None
                max_val = int(query_params.get(f"{col}_max")) if f"{col}_max" in query_params else None

                # IB logic
                if col.lower().startswith("ib"):
                    if r.get("12th Board", "").strip().upper() != "IBDP":
                        include = False
                        break
                    col = "12th grade overall score"

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break
            else:
                if key.lower() == "intended_major":
                    if not fuzzy_match(r.get("Intended Major"), value, multiple=True):
                        include = False
                        break
                elif key.lower() == "city":
                    if value.lower() != str(r.get("City of Graduation", "")).lower():
                        include = False
                        break
                elif key.lower() in ["countries applied to", "admitted univs", "rejected univs"]:
                    if not fuzzy_match(r.get(key), value):
                        include = False
                        break
                else:
                    if not fuzzy_match(r.get(key), value):
                        include = False
                        break

        if include:
            filtered.append(r)

    return {
        "count": len(filtered),
        "students": filtered
    }


# -------------------------------
# Phase 2 – Natural Language Query
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: NLQueryRequest):
    system_prompt = """
You convert user queries into filter JSON for a students API.

Rules:
- Output ONLY valid JSON
- No explanation
- Use keys exactly as API expects
- SAT and ACT are mutually exclusive
- IB filters use keys like ib_min_12, ib_max_12
- Intended majors should be comma-separated if multiple
"""

    completion = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.query}
        ]
    )

    raw_text = completion.choices[0].message.content

    try:
        filters = extract_json_from_llm(raw_text)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="LLM output could not be parsed"
        )

    return {
        "interpreted_filters": filters,
        "students_endpoint": "/students",
        "query_params": filters
    }
