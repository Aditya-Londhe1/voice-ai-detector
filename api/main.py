from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import os

from api.detector import VoiceDetector


API_KEY = os.getenv("API_KEY", "default_key")

ALLOWED_LANGS = ["ta", "en", "hi", "ml", "te"]


app = FastAPI(title="Voice AI Detector", version="1.0")

detector = VoiceDetector()


# ---------------- SCHEMA ----------------

class DetectRequest(BaseModel):
    audio_base64: str = Field(..., alias="audioBase64")
    language: str

    class Config:
        populate_by_name = True


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

    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API Key")

    if data.language not in ALLOWED_LANGS:
        raise HTTPException(400, "Unsupported language")

    try:
        prob = detector.predict(data.audio_base64)

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(400, "Invalid audio format")

    return {
        "result": "AI_GENERATED" if prob > 0.5 else "HUMAN",
        "confidence": round(prob, 3),
        "language": data.language
    }


@app.get("/")
def root():
    return {"status": "running"}
