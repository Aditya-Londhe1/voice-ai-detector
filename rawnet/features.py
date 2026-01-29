import librosa
import numpy as np


MAX_LEN = 400   # number of frames


def extract_mfcc(audio, sr=16000, n_mfcc=40):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256
    )

    # Normalize
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

    # Pad / Trim
    if mfcc.shape[1] < MAX_LEN:

        pad_width = MAX_LEN - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc
