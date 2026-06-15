import sys
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, correlate, correlation_lags  # pyright: ignore[reportMissingImports]
from subprocess import run, PIPE
from dataclasses import dataclass

__version__ = "0.2.1"


@dataclass
class Spec:
    frequencies: np.ndarray
    times: np.ndarray
    spectrogram: np.ndarray


@dataclass
class Result:
    error_matrix: np.ndarray
    score: float
    frequency_delta: float
    time_delta: float


def check_ffmpeg(ffmpeg_exe="ffmpeg"):
    if not shutil.which(ffmpeg_exe):
        print(
            f"Error: '{ffmpeg_exe}' executable not found. Provide --ffmpeg-path or install FFmpeg.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def load(path, sample_rate=48000, channels=1, ffmpeg_exe="ffmpeg"):
    process = run(
        [ffmpeg_exe, "-v", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(sample_rate), "-"],
        stdout=PIPE, stderr=PIPE
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode())
    
    audio_data = np.frombuffer(process.stdout, np.float32)
    if channels > 1:
        audio_data = audio_data.reshape(-1, channels)
    return audio_data


def normalize(audio_data):
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 1e-9:
        return audio_data / max_amplitude
    return audio_data


def align_signals(audio_a, audio_b, sample_rate):
    # Use mono mix or first channel for correlation
    mono_a = audio_a[:, 0] if len(audio_a.shape) > 1 else audio_a
    mono_b = audio_b[:, 0] if len(audio_b.shape) > 1 else audio_b
    
    # Downsample for fast correlation (target ~4000 Hz)
    downsample_factor = max(1, sample_rate // 4000)
    downsampled_a = mono_a[::downsample_factor]
    downsampled_b = mono_b[::downsample_factor]
    
    # Restrict window to first 30 seconds for speed
    limit = int(30 * 4000)
    downsampled_a = downsampled_a[:limit]
    downsampled_b = downsampled_b[:limit]
    
    cross_correlation = correlate(downsampled_a, downsampled_b, mode='full', method='fft')
    lags = correlation_lags(len(downsampled_a), len(downsampled_b), mode='full')
    
    best_lag_downsampled = lags[np.argmax(np.abs(cross_correlation))]
    best_lag = int(best_lag_downsampled * downsample_factor)
    
    return best_lag


def spec(audio_data, sample_rate, nperseg=2048, noverlap=1536):
    if len(audio_data.shape) > 1:
        # Stereo / multi-channel
        spectrograms = []
        for channel in range(audio_data.shape[1]):
            frequencies, times, stft_matrix = stft(audio_data[:, channel], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            spectrogram = 20 * np.log10(np.abs(stft_matrix) + 1e-10)
            spectrograms.append(spectrogram)
        stacked_spectrograms = np.stack(spectrograms, axis=-1)
        return Spec(frequencies, times, stacked_spectrograms)
    else:
        frequencies, times, stft_matrix = stft(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        spectrogram = 20 * np.log10(np.abs(stft_matrix) + 1e-10)
        return Spec(frequencies, times, spectrogram)


def compare(spec_a, spec_b):
    error_matrix = np.abs(spec_a.spectrogram - spec_b.spectrogram)
    # Average across channels if multi-channel for summary score
    mean_error = np.mean(error_matrix, axis=-1) if len(error_matrix.shape) > 2 else error_matrix
    mean_spec_a = np.mean(np.abs(spec_a.spectrogram), axis=-1) if len(spec_a.spectrogram.shape) > 2 else np.abs(spec_a.spectrogram)
    
    score = np.clip(100 * (1 - np.mean(mean_error) / (np.mean(mean_spec_a) + 1e-9)), 0, 100)
    frequency_delta = np.mean(np.std(mean_error, 1))
    time_delta = np.mean(np.std(mean_error, 0))
    return Result(error_matrix, score, frequency_delta, time_delta)


def run_tool():

    
    parser = argparse.ArgumentParser(description="Compare original audio to a recreation.")
    parser.add_argument("original", help="Path to original audio file")
    parser.add_argument("recreation", help="Path to recreated audio file")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate (default: 48000)")
    parser.add_argument("--stereo", action="store_true", help="Compare in stereo (default: mono)")
    parser.add_argument("--align", action="store_true", help="Auto-align starting times using cross-correlation")
    parser.add_argument("--no-norm", action="store_true", help="Disable automatic volume normalization")
    parser.add_argument("--nperseg", type=int, default=2048, help="STFT window size (default: 2048)")
    parser.add_argument("--noverlap", type=int, default=1536, help="STFT overlap size (default: 1536)")
    parser.add_argument("-o", "--output", help="Save comparison plot to this file path")
    parser.add_argument("--ffmpeg-path", help="Path to ffmpeg executable (overrides PATH detection)")

    parser.add_argument("--headless", action="store_true", help="Run without opening interactive plot window")

    args = parser.parse_args()
    channels = 2 if args.stereo else 1
    
    # Determine ffmpeg executable
    ffmpeg_exe = args.ffmpeg_path if args.ffmpeg_path else shutil.which("ffmpeg")
    if not ffmpeg_exe:
        print("Error: ffmpeg executable not found. Provide --ffmpeg-path or ensure ffmpeg is on PATH.", file=sys.stderr)
        sys.exit(1)
    
    # Verify ffmpeg availability
    check_ffmpeg(ffmpeg_exe)
    
    # Load audio using determined ffmpeg
    original_audio = load(args.original, args.sr, channels, ffmpeg_exe)
    recreation_audio = load(args.recreation, args.sr, channels, ffmpeg_exe)
    
    # Auto align if requested
    if args.align:
        lag = align_signals(original_audio, recreation_audio, args.sr)
        if lag > 0:
            original_audio = original_audio[lag:]
        elif lag < 0:
            recreation_audio = recreation_audio[-lag:]
    
    # Clip to same length
    min_length = min(len(original_audio), len(recreation_audio))
    original_audio, recreation_audio = original_audio[:min_length], recreation_audio[:min_length]
    
    # Normalize if requested
    if not args.no_norm:
        original_audio = normalize(original_audio)
        recreation_audio = normalize(recreation_audio)
        
    spec_a = spec(original_audio, args.sr, args.nperseg, args.noverlap)
    spec_b = spec(recreation_audio, args.sr, args.nperseg, args.noverlap)
    comparison_result = compare(spec_a, spec_b)
    
    # --- STDOUT live summary ---
    print(f"\n=== AUDIO COMPARE v{__version__} ===")
    print(f"channels   : {'stereo' if args.stereo else 'mono'}")
    print(f"aligned    : {'yes' if args.align else 'no'}")
    print(f"normalized : {'yes' if not args.no_norm else 'no'}")
    print(f"score      : {comparison_result.score:.2f}")
    print(f"freq delta : {comparison_result.frequency_delta:.4f}")
    print(f"time delta : {comparison_result.time_delta:.4f}")
    print("===============================\n")
    
    # Pre-average spectrograms for visualization if stereo
    original_plot = np.mean(spec_a.spectrogram, axis=-1) if len(spec_a.spectrogram.shape) > 2 else spec_a.spectrogram
    recreation_plot = np.mean(spec_b.spectrogram, axis=-1) if len(spec_b.spectrogram.shape) > 2 else spec_b.spectrogram
    error_plot = np.mean(comparison_result.error_matrix, axis=-1) if len(comparison_result.error_matrix.shape) > 2 else comparison_result.error_matrix
    
    # --- plot ---
    plt.style.use("dark_background")
    fig = plt.figure(num="Spectrogram Comparison Tool", figsize=(14, 10))
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    
    ax1.set_title("Original")
    ax1.imshow(original_plot, aspect="auto", origin="lower")
    
    ax2.set_title("Recreated")
    ax2.imshow(recreation_plot, aspect="auto", origin="lower")
    
    ax3.set_title("Error (|A-B|)")
    im = ax3.imshow(error_plot, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax3)
    
    fig.suptitle(
        f"score={comparison_result.score:.2f} | fd={comparison_result.frequency_delta:.4f} | td={comparison_result.time_delta:.4f}"
    )
    
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Plot saved to: {args.output}")
        
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    run_tool()