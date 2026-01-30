import torch
import base64
import io
import numpy as np
from scipy.io import wavfile
import librosa
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rawnet.model import RawNetLite
from rawnet.features import extract_mfcc


class VoiceDetector:

    def __init__(self):

        torch.set_num_threads(1)

        self.device = "cpu"

        self.model = RawNetLite().to(self.device)

        self.model.load_state_dict(
            torch.load("models/rawnet_custom.pth", map_location="cpu")
        )

        self.model.eval()


    def decode_audio(self, base64_audio):

        base64_audio = base64_audio.strip()
        base64_audio = base64_audio.replace("\n", "").replace(" ", "")

        if "base64," in base64_audio:
            base64_audio = base64_audio.split("base64,")[1]

        # fix padding
        missing = len(base64_audio) % 4
        if missing:
            base64_audio += "=" * (4 - missing)

        audio_bytes = base64.b64decode(base64_audio)

        buffer = io.BytesIO(audio_bytes)

        sr, audio = wavfile.read(buffer)

        # normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.max(np.abs(audio))

        # stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # resample
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)

        # pad
        if len(audio) < 16000:
            audio = librosa.util.fix_length(audio, 16000)

        return audio


    def predict(self, base64_audio):

        audio = self.decode_audio(base64_audio)

        mfcc = extract_mfcc(audio)

        x = torch.tensor(mfcc).unsqueeze(0).float()

        with torch.no_grad():
            out = self.model(x)
            prob = torch.sigmoid(out).item()

        return prob
