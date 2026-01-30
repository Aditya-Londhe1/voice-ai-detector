import torch
import base64
import io
import numpy as np
import soundfile as sf
import librosa
from scipy.io import wavfile

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rawnet.model import RawNetLite
from rawnet.features import extract_mfcc


class VoiceDetector:

    def __init__(self):

        self.device = "cpu"

        self.model = RawNetLite().to(self.device)

        self.model.load_state_dict(
            torch.load(
                "models/rawnet_custom.pth",
                map_location=self.device
            )
        )

        self.model.eval()


    def decode_audio(self, base64_audio):

        if not isinstance(base64_audio, str):
            raise ValueError("Audio must be string")

        # Clean EVERYTHING
        base64_audio = base64_audio.strip()

        base64_audio = base64_audio.replace("\n", "")
        base64_audio = base64_audio.replace("\r", "")
        base64_audio = base64_audio.replace(" ", "")

        # Remove data URI
        if "base64," in base64_audio:
            base64_audio = base64_audio.split("base64,")[1]

        # Fix padding
        missing = len(base64_audio) % 4
        if missing:
            base64_audio += "=" * (4 - missing)

        try:
            audio_bytes = base64.b64decode(base64_audio, validate=False)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")

        if len(audio_bytes) < 1000:
            raise ValueError("Audio too small")

        buffer = io.BytesIO(audio_bytes)

        try:
            sr, audio = wavfile.read(buffer)
        except Exception as e:
            raise ValueError(f"WAV read failed: {e}")

        if audio is None:
            raise ValueError("No audio data")

        # Normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.max(np.abs(audio))

        # Stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

        # Pad
        if len(audio) < 16000:
            audio = librosa.util.fix_length(audio, 16000)

        return audio





    def predict(self, base64_audio):

        # Decode
        audio = self.decode_audio(base64_audio)


        # Fix length (at least 1 second)
        if len(audio) < 16000:
            audio = librosa.util.fix_length(audio, 16000)


        # Extract MFCC
        mfcc = extract_mfcc(audio)


        # To tensor
        x = torch.tensor(mfcc)\
                .unsqueeze(0)\
                .float()\
                .to(self.device)


        # Predict
        with torch.no_grad():

            out = self.model(x)

            prob = torch.sigmoid(out).item()


        return prob
