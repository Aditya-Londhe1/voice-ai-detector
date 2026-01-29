import base64
import requests


def encode_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


API_URL = "http://127.0.0.1:8000/detect"
API_KEY = "my_secret_key_123"   # Same as in main.py


audio_base64 = encode_audio(r"C:\Users\sai\Downloads\archive\real_samples\908_31957_000003_000001.wav")  # Put any wav here


payload = {
    "audio_base64": audio_base64,
    "language": "en"
}

headers = {
    "X-API-KEY": API_KEY
}


r = requests.post(API_URL, json=payload, headers=headers)

print(r.json())
