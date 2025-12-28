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

# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    params = dict(request.query_params)
    results = []

    # -------------------------------
    # SAT / ACT mutual exclusion
    # -------------------------------
    sat_used = any(k.startswith("SAT") for k in params)
    act_used = any(k.startswith("ACT") for k in params)

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        # -------------------------------
        # IB FILTER (locked)
        # -------------------------------
        if "ib_min_12" in params or "ib_max_12" in params:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            ib_score = to_int(r.get("12th grade overall score"))
            if ib_score is None:
                continue

            if "ib_min_12" in params and ib_score < int(params["ib_min_12"]):
                continue
            if "ib_max_12" in params and ib_score > int(params["ib_max_12"]):
                continue

        # -------------------------------
        # Numeric filters (SAT / ACT)
        # -------------------------------
        for k, v in params.items():
            if not (k.endswith("_min") or k.endswith("_max")):
                continue

            col = k.rsplit("_", 1)[0]
            if col.startswith("ib"):
                continue

            val = to_int(r.get(col))
            if val is None:
                include = False
                break

            if k.endswith("_min") and val < int(v):
                include = False
                break
            if k.endswith("_max") and val > int(v):
                include = False
                break

        if not include:
            continue

        # -------------------------------
        # TEXT FILTERS (FIXED)
        # -------------------------------
        for k, v in params.items():
            if k.endswith("_min") or k.endswith("_max"):
                continue

            key = k.lower()

            if key == "intended_major":
                majors = [m.strip() for m in v.split(",")]
                if not any(fuzzy_match(r.get("Intended Major", ""), m) for m in majors):
                    include = False
                    break

            elif key == "city":
                if r.get("City of Graduation", "").lower() != v.lower():
                    include = False
                    break

            elif key in ["admitted univs", "rejected univs", "countries applied to"]:
                sheet_col = TEXT_COLUMN_MAP[key]
                if not fuzzy_match(r.get(sheet_col, ""), v):
                    include = False
                    break

        if include:
            results.append(r)

    return {
        "count": len(results),
        "students": results
    }


@app.post("/nl_query")
async def nl_query(payload: NLQuery):
    """
    Temporary Phase-2 endpoint.
    Converts NL → filters (stubbed for now).
    """

    user_query = payload.query.lower()

    # 🔹 TEMP RULE-BASED PARSER (no LLM yet)
    filters = {}

    if "ib" in user_query:
        filters["ib_min_12"] = 35

    if "georgia" in user_query:
        filters["admitted univs"] = "Georgia"

    if "computer science" in user_query:
        filters["intended_major"] = "Computer Science"

    return {
        "interpreted_filters": filters,
        "note": "LLM not wired yet — rule based Phase 2.1"
    }
