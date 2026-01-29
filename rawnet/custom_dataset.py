import os
import torch
import librosa

from features import extract_mfcc


class CustomDeepfakeDataset(torch.utils.data.Dataset):

    def __init__(self, root):

        self.data = []

        for folder in os.listdir(root):

            folder_path = os.path.join(root, folder)

            if not os.path.isdir(folder_path):
                continue

            # Human
            if folder == "real_samples":
                label = 1

            # AI
            else:
                label = 0

            for file in os.listdir(folder_path):

                if file.endswith(".wav") or file.endswith(".mp3"):

                    self.data.append((
                        os.path.join(folder_path, file),
                        label
                    ))

        print("Total samples:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path, label = self.data[idx]

        audio, _ = librosa.load(path, sr=16000)

        if len(audio) < 16000:
            audio = librosa.util.fix_length(audio, 16000)

        mfcc = extract_mfcc(audio)

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
