import torch
import base64
import io
import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf
import tempfile
import uuid
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

        try:
            audio_bytes = base64.b64decode(base64_audio)
        except Exception:
             # invalid base64
            return np.zeros(16000, dtype=np.float32)

        # Write to temp file for librosa to load (handles mp3, wav, etc.)
        # suffix .mp3 helps librosa/soundfile guess format if header is ambiguous,
        # but usually header detection works. Let's use a generic name or try to detect?
        # MP3 input is expected.
        path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + ".mp3")
        
        with open(path, "wb") as f:
            f.write(audio_bytes)

        try:
            # sr=16000 ensures resampling happens during load
            audio, _ = librosa.load(path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading audio: {e}")
            if os.path.exists(path):
                os.remove(path)
            return np.zeros(16000, dtype=np.float32)
        
        if os.path.exists(path):
            os.remove(path)

        # pad if too short
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
