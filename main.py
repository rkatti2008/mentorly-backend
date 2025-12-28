import os
import json
import urllib.parse
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
from openai import OpenAI

app = FastAPI()

# =====================================================
# Google Sheets Setup
# =====================================================
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Service account from environment variable (Render-safe)
service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])

creds = Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES
)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1


# =====================================================
# OpenAI Client
# =====================================================
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# Helper Functions
# =====================================================
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


def fuzzy_match(text, patterns, threshold=0.7):
    if patterns is None:
        return True
    if text is None:
        return False

    text = str(text).lower()
    patterns = [p.strip().lower() for p in str(patterns).split(",")]

    for pattern in patterns:
        if pattern in text:
            return True
        if SequenceMatcher(None, text, pattern).ratio() >= threshold:
            return True

    return False


# =====================================================
# Filter whitelist (CRITICAL SAFETY)
# =====================================================
ALLOWED_FILTERS = {
    "city": "City of Graduation",
    "intended_major": "Intended Major",
    "sat_min": "SAT Total score",
    "sat_max": "SAT Total score",
    "act_min": "ACT Score",
    "act_max": "ACT Score",
    "ib_min_12": "12th grade overall score",
    "ib_max_12": "12th grade overall score",
    "countries applied to": "Countries Applied To",
    "admitted univs": "Admitted Univs",
    "rejected univs": "Rejected Univs"
}


# =====================================================
# /students endpoint (AUTHORITATIVE)
# =====================================================
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    filtered = []

    query_params = dict(request.query_params)

    # SAT vs ACT mutual exclusion
    sat_used = any(k.startswith("sat_") for k in query_params)
    act_used = any(k.startswith("act_") for k in query_params)

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        for key, value in query_params.items():
            # -------------------------
            # Numeric filters
            # -------------------------
            if key.endswith("_min") or key.endswith("_max"):
                base = key.rsplit("_", 1)[0]

                min_val = query_params.get(f"{base}_min")
                max_val = query_params.get(f"{base}_max")

                min_val = int(min_val) if min_val is not None else None
                max_val = int(max_val) if max_val is not None else None

                # IB logic
                if base.startswith("ib"):
                    if r.get("12th Board", "").strip().upper() != "IBDP":
                        include = False
                        break
                    col = "12th grade overall score"
                else:
                    col = ALLOWED_FILTERS.get(key) or ALLOWED_FILTERS.get(base)

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break

            # -------------------------
            # Text filters
            # -------------------------
            else:
                if key == "city":
                    if value.lower() != str(r.get("City of Graduation", "")).lower():
                        include = False
                        break
                else:
                    col = ALLOWED_FILTERS.get(key)
                    if col and not fuzzy_match(r.get(col), value):
                        include = False
                        break

        if include:
            filtered.append(r)

    return {
        "count": len(filtered),
        "students": filtered
    }


# =====================================================
# Natural Language → Query Builder
# =====================================================
class NLQuery(BaseModel):
    query: str


@app.post("/nl_query")
async def nl_query(req: NLQuery):
    """
    Converts natural language to structured /students filters
    """

    prompt = f"""
You are a STRICT query parser.

Convert the user query into VALID JSON using ONLY the allowed filters.
Rules:
- Use only keys from the allowed list
- SAT and ACT are mutually exclusive
- IB filters apply only to IBDP students
- Use integers for numeric filters
- Use comma-separated strings for text filters
- Output ONLY JSON

Allowed filters:
{list(ALLOWED_FILTERS.keys())}

User query:
"{req.query}"
"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        filters = json.loads(response.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=400, detail="LLM output could not be parsed")

    # Enforce whitelist
    filters = {k: v for k, v in filters.items() if k in ALLOWED_FILTERS}

    # Call students logic internally
    request_scope = Request(
        scope={
            "type": "http",
            "query_string": urllib.parse.urlencode(filters).encode()
        }
    )

    result = await get_students(request_scope)

    return {
        "interpreted_filters": filters,
        "generated_query": f"/students?{urllib.parse.urlencode(filters)}",
        "count": result["count"],
        "students": result["students"]
    }
