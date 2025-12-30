from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json

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
sheet = client.open_by_key(SHEET_ID).sheet1

client_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")

# -------------------------------
# Phase 3.1 — Canonical maps
# -------------------------------
COUNTRY_CANONICAL = {
    "america": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "us": "usa",
    "amrika": "usa",
    "u.s.": "usa",

    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "england": "united kingdom",
}

MAJOR_CANONICAL = {
    "cs": "computer science",
    "comp sci": "computer science",
    "computer sciences": "computer science",

    "ai": "artificial intelligence",
    "artificial intel": "artificial intelligence",
}

UNIV_CANONICAL = {
    "gatech": "georgia tech",
    "georgia institute of technology": "georgia tech",
}

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

    sat_used = any(k.startswith("SAT") for k in query_params)
    act_used = any(k.startswith("ACT") for k in query_params)

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
# Phase 3.4.3 — Repair LLM mistakes
# -------------------------------
def repair_llm_filters(filters: dict) -> dict:
    repaired = dict(filters)

    COUNTRY_WORDS = {"america", "usa", "us", "united states", "uk", "canada", "india"}

    if "admitted univs" in repaired:
        val = str(repaired["admitted univs"]).lower().strip()
        if val in COUNTRY_WORDS:
            repaired.pop("admitted univs")
            repaired["countries applied to"] = val

    if "ib_min_12" in repaired and "ib" not in repaired:
        repaired.pop("ib_min_12")

    return repaired

# -------------------------------
# Phase 5.1 — RAG Prompt Builder (NEW)
# -------------------------------
def build_rag_messages(user_question: str, rows: list):
    system_prompt = """
You are Mentorly, a college counseling assistant.

You are given:
1) A user question
2) A set of student records retrieved from a database

CRITICAL RULES:
- You MUST use ONLY the provided student records.
- You MUST NOT use any external knowledge, assumptions, or general advice.
- If the provided records do not contain enough information to answer the question, say:
  "Based on the available data, there is not enough information to answer this question."

ALLOWED:
- Summarizing patterns visible in the records
- Counting, grouping, and comparing the records
- Rephrasing the data in clear natural language

FORBIDDEN:
- Predicting outcomes not shown in the data
- Recommending universities or strategies not present in the records
- Giving advice beyond what the data directly supports

OUTPUT FORMAT:
- A concise paragraph (3–6 sentences)
- Reference only information visible in the records
"""

    user_prompt = f"""
User question:
{user_question}

Retrieved student records:
{json.dumps(rows, indent=2)}

Number of matching records:
{len(rows)}
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

# -------------------------------
# POST /nl_query
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):

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

Rules for IB:
- Phrases like "IB students" or "IBDP students" mean the student board is IB
- If no IB score is mentioned, emit ib_min_12 = 1

User query:
"{req.message}"
"""

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()
    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1:
        raise HTTPException(status_code=400, detail="LLM output did not contain JSON")

    filters = json.loads(raw_output[start:end + 1])
    filters = repair_llm_filters(filters)
    filters = normalize_filters(filters)

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    # -------------------------------
    # Phase 5.2 — RAG Answer Synthesis
    # -------------------------------
    if len(students) == 0:
        assistant_answer = "No matching student records were found for your query."
    else:
        rag_messages = build_rag_messages(req.message, students)
        rag_response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=rag_messages,
            temperature=0
        )
        assistant_answer = rag_response.choices[0].message.content.strip()

    return {
        "interpreted_filters": filters,
        "count": len(students),
        "assistant_answer": assistant_answer,
        "students": students
    }
