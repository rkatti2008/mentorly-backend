from fastapi import FastAPI, Request, HTTPException
import gspread, os, json
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
from pydantic import BaseModel




app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SHEET_ID = os.environ.get("SHEET_ID")
if not SHEET_ID:
    raise RuntimeError("SHEET_ID not set")

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1


class NLQuery(BaseModel):
    query: str

class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Helpers
# -------------------------------
def to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def fuzzy_match(text, query, threshold=0.6):
    if not text or not query:
        return False

    text = text.lower()
    query = query.lower()

    if query in text:
        return True

    return SequenceMatcher(None, text, query).ratio() >= threshold

# -------------------------------
# Query → Sheet column mapping
# -------------------------------
TEXT_COLUMN_MAP = {
    "admitted univs": "Admitted Univs",
    "rejected univs": "Rejected Univs",
    "countries applied to": "Countries Applied To",
    "intended_major": "Intended Major",
    "city": "City of Graduation",
}


def filter_students(records, query_params):
    filtered = []

    # Detect SAT and ACT range filters
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
                else:
                    if not fuzzy_match(r.get(key), value):
                        include = False
                        break

        if include:
            filtered.append(r)

    return filtered




# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    query_params = dict(request.query_params)

    filtered = filter_students(records, query_params)

    return {
        "count": len(filtered),
        "students": filtered
    }

def call_llm(prompt: str) -> str:
    """
    TEMPORARY stub for Phase 2.2.
    This will be replaced with real LLM integration later.
    """

    # Very naive rule-based fallback (for now)
    if "Georgia" in prompt:
        return json.dumps({
            "filters": {
                "admitted univs": "Georgia"
            }
        })

    return json.dumps({"filters": {}})


def llm_to_filters(nl_query: str) -> dict:
    """
    Calls LLM and converts NL → structured filters.
    """

    prompt = f"""
You are an API that converts student search queries into JSON filters.

RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No extra text

Allowed fields:
ib_min_12, ib_max_12, sat_min, sat_max, act_min, act_max,
admitted univs, intended_major

Query:
"{nl_query}"

Return format:
{{
  "filters": {{
    ...
  }}
}}
"""

    #  CALL YOUR LLM HERE (example placeholder)
    llm_response = call_llm(prompt)   # ← your existing LLM call

    try:
        parsed = json.loads(llm_response)
        return parsed.get("filters", {})
    except Exception:
        raise ValueError("LLM output could not be parsed")


@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    nl = req.message.lower()

    # Phase 2.1/2.2 logic (existing)
    filters = {}

    if "ib" in nl and "35" in nl:
        filters["ib_min_12"] = 35
    if "georgia" in nl:
        filters["admitted univs"] = "Georgia"
    if "computer" in nl:
        filters["intended_major"] = "Computer Science"

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    return {
        "interpreted_filters": filters,
        "count": len(students),
        "students": students
    }
