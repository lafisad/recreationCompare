import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from subprocess import run, PIPE
from dataclasses import dataclass


__version__ = "0.1.7-260614"


@dataclass
class Spec:
    f: np.ndarray
    t: np.ndarray
    s: np.ndarray


@dataclass
class Result:
    err: np.ndarray
    score: float
    fd: float
    td: float


def load(path, sr=48000):
    p = run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(sr), "-"],
        stdout=PIPE, stderr=PIPE
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    return np.frombuffer(p.stdout, np.float32)


def spec(x, sr):
    f, t, z = stft(x, fs=sr, nperseg=2048, noverlap=1536)
    s = 20 * np.log10(np.abs(z) + 1e-10)
    return Spec(f, t, s)


def compare(a, b):
    e = np.abs(a.s - b.s)
    score = np.clip(100 * (1 - np.mean(e) / (np.mean(np.abs(a.s)) + 1e-9)), 0, 100)
    return Result(e, score, np.mean(np.std(e, 1)), np.mean(np.std(e, 0)))


def run_tool(a_path, b_path):
    sr = 48000

    a = load(a_path, sr)
    b = load(b_path, sr)

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    A = spec(a, sr)
    B = spec(b, sr)
    R = compare(A, B)

    # --- STDOUT live summary ---
    print(f"\n=== AUDIO COMPARE v{__version__} ===")
    print(f"score      : {R.score:.2f}")
    print(f"freq delta : {R.fd:.4f}")
    print(f"time delta : {R.td:.4f}")
    print("===============================\n")

    # --- plot ---
    fig = plt.figure(num="Spectrogram Comparison Tool", figsize=(14, 10))
    plt.style.use("dark_background")

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.set_title("Original")
    ax1.imshow(A.s, aspect="auto", origin="lower")

    ax2.set_title("Recreated")
    ax2.imshow(B.s, aspect="auto", origin="lower")

    ax3.set_title("Error (|A-B|)")
    im = ax3.imshow(R.err, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax3)

    fig.suptitle(
        f"score={R.score:.2f} | fd={R.fd:.4f} | td={R.td:.4f}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare original.mp3 recreated.mp3")
        sys.exit(1)

    run_tool(sys.argv[1], sys.argv[2])