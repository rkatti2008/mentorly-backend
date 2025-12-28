from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json

app = FastAPI()

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

client_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")

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
    """
    Case-insensitive + space-insensitive column lookup
    Fixes fuzzy filter bugs for admitted univs, countries applied to, etc.
    """
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

    # SAT / ACT mutual exclusion
    sat_used = any(k.startswith("SAT") for k in query_params)
    act_used = any(k.startswith("ACT") for k in query_params)

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        # ---- IB FILTER ----
        if "ib_min_12" in query_params or "ib_max_12" in query_params:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            min_ib = query_params.get("ib_min_12")
            max_ib = query_params.get("ib_max_12")

            try:
                min_ib = int(min_ib) if min_ib is not None else None
                max_ib = int(max_ib) if max_ib is not None else None
            except ValueError:
                continue

            if not passes_numeric_filter(
                r.get("12th grade overall score"),
                min_ib,
                max_ib,
            ):
                continue

        # ---- SAT FILTER (FIXED) ----
        if "SAT Total score_min" in query_params or "SAT Total score_max" in query_params:
            if not passes_numeric_filter(
                r.get("SAT Total score"),
                query_params.get("SAT Total score_min"),
                query_params.get("SAT Total score_max"),
            ):
                continue

        # ---- ACT FILTER ----
        if "ACT Score_min" in query_params or "ACT Score_max" in query_params:
            if not passes_numeric_filter(
                r.get("ACT Score"),
                query_params.get("ACT Score_min"),
                query_params.get("ACT Score_max"),
            ):
                continue

        # ---- TEXT FILTERS ----
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
# GET /students
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    query_params = dict(request.query_params)

    # Convert numeric params
    for k in list(query_params.keys()):
        if k.endswith("_min") or k.endswith("_max"):
            try:
                query_params[k] = int(query_params[k])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric value for {k}"
                )

    students = filter_students(records, query_params)

    return {
        "count": len(students),
        "students": students
    }



@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    """
    Natural language → LLM → filters → students
    """

prompt = f"""
You are a strict JSON generator.

Your task:
Convert the user query into a JSON object of filters.

CRITICAL RULES:
- Output ONLY raw JSON
- Do NOT include markdown
- Do NOT include backticks
- Do NOT include explanations
- Do NOT include text before or after JSON
- JSON must start with {{ and end with }}

Allowed keys ONLY:
ib_min_12, ib_max_12,
"SAT Total score_min", "SAT Total score_max",
"ACT Score_min", "ACT Score_max",
intended_major,
admitted univs,
countries applied to,
city

Rules:
- SAT and ACT must NEVER both appear
- Use numbers for numeric values
- Use strings for text values
- Omit keys not mentioned in the query

User query:
"{req.message}"
"""
  

    try:
        response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()
        
        # ---- SANITIZE LLM OUTPUT ----
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            raw_output = raw_output.replace("json", "", 1).strip()
            
            
        # Remove accidental leading text
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        
        if start == -1 or end == -1:
            raise HTTPException(
                status_code=400,
                detail="LLM output did not contain JSON"
            )
        json_str = raw_output[start:end + 1]
        
        try:
            filters = json.loads(json_str)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="LLM output could not be parsed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # -------------------------------
    # Execute filters safely
    # -------------------------------
    records = sheet.get_all_records()
    students = filter_students(records, filters)

    return {
        "interpreted_filters": filters,
        "count": len(students),
        "students": students
    }
