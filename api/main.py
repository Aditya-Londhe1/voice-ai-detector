from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from api.detector import VoiceDetector


# ---------------- CONFIG ----------------

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

    # Support both formats
    audio_base64: Optional[str] = None   # snake_case
    audioBase64: Optional[str] = None    # camelCase (tester)
    audioFormat: Optional[str] = None

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

    # -------- AUTH --------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


    # -------- LANGUAGE CHECK --------
    if data.language not in ALLOWED_LANGS:
        raise HTTPException(status_code=400, detail="Unsupported language")


    # -------- GET AUDIO --------
    audio_b64 = data.audio_base64 or data.audioBase64

    if not audio_b64:
        raise HTTPException(status_code=400, detail="No audio provided")


    # -------- PREDICT --------
    try:
        prob = detector.predict(audio_b64)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio format")


    # -------- RESULT --------
    if prob > 0.5:
        result = "AI_GENERATED"
    else:
        result = "HUMAN"


    return {
        "result": result,
        "confidence": round(float(prob), 3),
        "language": data.language
    }


# ---------------- HEALTH ----------------

@app.get("/")
def root():
    return {"status": "running"}
