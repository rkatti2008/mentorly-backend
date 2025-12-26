from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add this block
origins = ["*"]  # Allow all origins for now (simplest)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # <- frontend domains can be added here instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": f"Chuchu, the Master of Universe, says: {req.message}"}
