import torch
import librosa
import base64
import io
import numpy as np

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

        audio_bytes = base64.b64decode(base64_audio)

        buffer = io.BytesIO(audio_bytes)

        audio, _ = librosa.load(buffer, sr=16000)

        return audio


    def predict(self, base64_audio):

        audio = self.decode_audio(base64_audio)

        # Fix length
        if len(audio) < 16000:
            audio = librosa.util.fix_length(audio, 16000)

        # Extract MFCC
        mfcc = extract_mfcc(audio)

        x = torch.tensor(mfcc)\
                .unsqueeze(0)\
                .float()\
                .to(self.device)

        with torch.no_grad():

            out = self.model(x)

            prob = torch.sigmoid(out).item()

        return prob
