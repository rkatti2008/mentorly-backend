from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread, os, json
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

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

# -------------------------------
# Helpers
# -------------------------------
def to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def fuzzy_match(text, query, threshold=0.65):
    if not text or not query:
        return False
    text = text.lower()
    query = query.lower()

    # Strong substring check first
    if query in text:
        return True

    return SequenceMatcher(None, text, query).ratio() >= threshold

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
        # IB FILTER (LOCKED)
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
                continue  # already handled above

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
        # TEXT FILTERS
        # -------------------------------
        for k, v in params.items():
            if k.endswith("_min") or k.endswith("_max"):
                continue

            # Intended Major (multi-value)
            if k.lower() == "intended_major":
                majors = [m.strip() for m in v.split(",")]
                if not any(fuzzy_match(r.get("Intended Major", ""), m) for m in majors):
                    include = False
                    break

            # City (exact)
            elif k.lower() == "city":
                if r.get("City of Graduation", "").lower() != v.lower():
                    include = False
                    break

            # Universities (LOCKED fuzzy logic)
            elif k.lower() in ["admitted univs", "rejected univs", "countries applied to"]:
                if not fuzzy_match(r.get(k, ""), v):
                    include = False
                    break

        if include:
            results.append(r)

    return {
        "count": len(results),
        "students": results
    }
