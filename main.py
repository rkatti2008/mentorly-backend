from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import json
import os

app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable not set")

creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# -------------------------------
# Pydantic model (optional future use)
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

# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    """
    Dynamic filtering endpoint with:
    - Numeric min/max filters (SAT, ACT, IB)
    - Multiple intended majors (comma-separated)
    - Fuzzy text matching
    - SAT and ACT mutually exclusive
    """

    records = sheet.get_all_records()
    filtered = []

    qp = dict(request.query_params)

    # -------------------------------
    # SAT vs ACT mutual exclusion
    # -------------------------------
    sat_columns = ["SAT Total score", "SAT Math", "SAT English"]
    act_columns = ["ACT Score"]

    sat_present = any(
        key.rsplit("_", 1)[0] in sat_columns
        for key in qp
        if key.endswith("_min") or key.endswith("_max")
    )

    act_present = any(
        key.rsplit("_", 1)[0] in act_columns
        for key in qp
        if key.endswith("_min") or key.endswith("_max")
    )

    if sat_present and act_present:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    # -------------------------------
    # Apply filters
    # -------------------------------
    for r in records:
        include = True

        for key, value in qp.items():

            # ---------- Numeric filters ----------
            if key.endswith("_min") or key.endswith("_max"):
                col = key.rsplit("_", 1)[0]
                min_val = int(qp.get(f"{col}_min")) if f"{col}_min" in qp else None
                max_val = int(qp.get(f"{col}_max")) if f"{col}_max" in qp else None

                # IB filter → only for IBDP students
                if col.lower().startswith("ib"):
                    if r.get("12th Board", "").strip().upper() != "IBDP":
                        include = False
                        break
                    col = "12th grade overall score"

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break

            # ---------- Text / fuzzy filters ----------
            else:
                key_l = key.lower()

                if key_l == "intended_major":
                    if not fuzzy_match(r.get("Intended Major"), value):
                        include = False
                        break

                elif key_l == "city":
                    if value.lower() != str(r.get("City of Graduation", "")).lower():
                        include = False
                        break

                elif key_l in ["countries applied to", "admitted univs", "rejected univs"]:
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
