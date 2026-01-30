from fastapi import FastAPI, Header, HTTPException, Request
import os

from api.detector import VoiceDetector


API_KEY = os.getenv("API_KEY", "default_key")

ALLOWED_LANGS = ["ta", "en", "hi", "ml", "te"]


app = FastAPI(title="Voice AI Detector", version="1.0")

# load model once
detector = VoiceDetector()


@app.post("/detect")
async def detect(request: Request, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

    body = await request.json()

    # accept both names
    audio = body.get("audio_base64") or body.get("audioBase64")
    lang = body.get("language")

    if not audio:
        raise HTTPException(400, "audio missing")

    if not lang:
        raise HTTPException(400, "language missing")

    if lang not in ALLOWED_LANGS:
        raise HTTPException(400, "unsupported language")

    try:
        prob = detector.predict(audio)
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(400, "Invalid audio")

    return {
        "result": "AI_GENERATED" if prob > 0.5 else "HUMAN",
        "confidence": round(prob, 3),
        "language": lang
    }


@app.get("/")
def root():
    return {"status": "running"}
