from fastapi import FastAPI, Query, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SERVICE_ACCOUNT_FILE = "service_account.json"
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1  # first worksheet

# -------------------------------
# Pydantic model for chat (if needed)
# -------------------------------
class ChatRequest(BaseModel):
    message: str

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
    """
    Return True if text approximately matches any of the patterns.
    Case-insensitive, partial match.
    - multiple=True for comma-separated pattern values
    """
    if patterns is None:
        return True
    if text is None:
        return False

    text = str(text).lower()
    if multiple or isinstance(patterns, str):
        patterns = [p.strip() for p in str(patterns).split(",")]

    for pattern in patterns:
        pattern = pattern.lower()
        if pattern in text:
            return True
        if SequenceMatcher(None, text, pattern).ratio() >= threshold:
            return True

    return False

# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    """
    Dynamic filtering endpoint with:
    - Fuzzy text search for text columns
    - Numeric min/max for SAT, ACT, IB
    - Multiple intended majors (comma-separated)
    - SAT and ACT filters are mutually exclusive
    - Fuzzy search for Countries Applied To, Admitted Univs, Rejected Univs
    """
    records = sheet.get_all_records()
    filtered = []

    query_params = dict(request.query_params)

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
                # Numeric filter
                col = key.rsplit("_", 1)[0]
                min_val = int(query_params.get(f"{col}_min")) if f"{col}_min" in query_params else None
                max_val = int(query_params.get(f"{col}_max")) if f"{col}_max" in query_params else None

                # Map IB filter to actual sheet column
                if col.lower().startswith("ib"):
                    if r.get("12th Board", "").strip().upper() != "IBDP":
                        include = False
                        break
                    col = "12th grade overall score"

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break
            else:
                # Text / fuzzy filter
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
