import torch
import base64
import io
import numpy as np
import soundfile as sf
import librosa

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

        # Remove spaces and newlines
        base64_audio = base64_audio.strip().replace("\n", "").replace(" ", "")

        # Remove data URI prefix if present
        if base64_audio.startswith("data:"):
            base64_audio = base64_audio.split(",")[1]

        # Fix padding if missing
        missing_padding = len(base64_audio) % 4
        if missing_padding:
            base64_audio += "=" * (4 - missing_padding)

        try:
            # Decode safely
            audio_bytes = base64.b64decode(base64_audio, validate=False)

        except Exception:
            raise ValueError("Base64 decode failed")

        buffer = io.BytesIO(audio_bytes)

        try:
            audio, sr = sf.read(buffer)

        except Exception:
            raise ValueError("Cannot decode WAV file")

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000
            )

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
