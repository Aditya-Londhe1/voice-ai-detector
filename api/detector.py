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
        torch.set_num_threads(1)


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

        

        # Clean base64
        base64_audio = base64_audio.strip().replace("\n", "").replace(" ", "")

        if base64_audio.startswith("data:"):
            base64_audio = base64_audio.split(",")[1]

        # Decode
        audio_bytes = base64.b64decode(base64_audio)

        buffer = io.BytesIO(audio_bytes)

        # Read wav using scipy (NO ffmpeg needed)
        sr, audio = wavfile.read(buffer)

        # Convert to float
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # Stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

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
