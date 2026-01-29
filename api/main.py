from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from api.detector import VoiceDetector



# ---------------- CONFIG ----------------

import os

API_KEY = os.getenv("API_KEY", "default_key")


ALLOWED_LANGS = ["ta", "en", "hi", "ml", "te"]


# ---------------- INIT ----------------

app = FastAPI(
    title="Voice AI Detector",
    version="1.0"
)

detector = VoiceDetector()


# ---------------- SCHEMA ----------------

class DetectRequest(BaseModel):

    audio_base64: str
    language: str


class DetectResponse(BaseModel):

    result: str
    confidence: float
    language: str


# ---------------- API ----------------

@app.post("/detect", response_model=DetectResponse)
def detect_voice(
    data: DetectRequest,
    x_api_key: str = Header(None)
):

    # Auth
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API Key")

    # Language check
    if data.language not in ALLOWED_LANGS:
        raise HTTPException(400, "Unsupported language")

    try:
        prob = detector.predict(data.audio_base64)

    except Exception as e:
        raise HTTPException(400, "Invalid audio format")


    # Result
    if prob > 0.5:
        result = "AI_GENERATED"
    else:
        result = "HUMAN"


    return {
        "result": result,
        "confidence": round(prob, 3),
        "language": data.language
    }


# ---------------- HEALTH ----------------

@app.get("/")
def root():
    return {"status": "running"}
