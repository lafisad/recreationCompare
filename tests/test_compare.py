import sys
import os
import unittest
import shutil
import random

# Graceful checks for dependencies
try:
    import numpy as np
    # pyrefly: ignore [missing-import]
    from scipy.io import wavfile
    # Add root folder to sys.path to import compare
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import compare
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR_MSG = str(e)


@unittest.skipIf(not HAS_DEPENDENCIES, f"Dependencies missing: {IMPORT_ERROR_MSG if 'IMPORT_ERROR_MSG' in locals() else 'numpy/scipy'}")
class TestAudioCompare(unittest.TestCase):
    
    def setUp(self):
        self.test_files = []
        self.ffmpeg_available = shutil.which("ffmpeg") is not None
        
    def tearDown(self):
        for file_path in self.test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    def generate_random_audio(self, filename, duration=1.5, sample_rate=48000, frequency=None, volume=1.0, delay_seconds=0.0):
        """Generates a slightly randomized/different sine wave file each time."""
        if frequency is None:
            frequency = random.uniform(300.0, 800.0)
            
        total_samples = int(sample_rate * duration)
        time_steps = np.linspace(0, duration, total_samples, endpoint=False)
        
        # Add a small amount of random noise to make each run unique
        noise = np.random.normal(0, 0.005, total_samples)
        audio = np.sin(2 * np.pi * frequency * time_steps) * volume + noise
        
        # Apply delay if specified
        if delay_seconds > 0:
            delay_samples = int(sample_rate * delay_seconds)
            delayed_audio = np.zeros_like(audio)
            if delay_samples < len(audio):
                delayed_audio[delay_samples:] = audio[:-delay_samples]
            audio = delayed_audio
            
        wavfile.write(filename, sample_rate, audio.astype(np.float32))
        self.test_files.append(filename)
        return frequency

    def test_normalize(self):
        # Generate raw array
        audio = np.array([0.1, -0.5, 0.3, -0.2], dtype=np.float32)
        normalized = compare.normalize(audio)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0)
        
        # Test zero signal
        zero_audio = np.zeros(10, dtype=np.float32)
        normalized_zero = compare.normalize(zero_audio)
        np.testing.assert_array_equal(normalized_zero, zero_audio)

    def test_align_signals(self):
        sample_rate = 48000
        duration = 1.0
        
        # Generate signals
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_a = np.sin(2 * np.pi * 440 * t)
        
        # Shift b by 0.1 seconds (4800 samples)
        shift_samples = 4800
        audio_b = np.zeros_like(audio_a)
        audio_b[shift_samples:] = audio_a[:-shift_samples]
        
        # Check alignment logic (b is delayed, so lag should be negative)
        detected_lag = compare.align_signals(audio_a, audio_b, sample_rate)
        # Allow small deviation due to downsampling resolution (factor of 12)
        self.assertTrue(abs(detected_lag - (-shift_samples)) <= 12)

    def test_compare_score(self):
        frequencies = np.linspace(0, 24000, 1025)
        times = np.linspace(0, 1.5, 80)
        
        # Create identical spectrograms
        spectrogram_a = np.random.uniform(-100, 0, (1025, 80))
        spec_a = compare.Spec(frequencies, times, spectrogram_a)
        spec_b = compare.Spec(frequencies, times, spectrogram_a)
        
        result = compare.compare(spec_a, spec_b)
        self.assertAlmostEqual(result.score, 100.0)

    def test_full_flow_with_ffmpeg(self):
        if not self.ffmpeg_available:
            self.skipTest("FFmpeg is not installed or not in PATH, skipping integration test.")
            
        file_a = "test_temp_a.wav"
        file_b = "test_temp_b.wav"
        
        # Generate different randomized frequency each run
        freq = self.generate_random_audio(file_a, duration=1.5, volume=1.0)
        self.generate_random_audio(file_b, duration=1.5, volume=0.5, frequency=freq, delay_seconds=0.05)
        
        # Load
        original_audio = compare.load(file_a, sample_rate=48000, channels=1)
        recreation_audio = compare.load(file_b, sample_rate=48000, channels=1)
        
        # Check normalization
        norm_a = compare.normalize(original_audio)
        norm_b = compare.normalize(recreation_audio)
        
        # Check alignment (b is delayed, lag should be negative)
        lag = compare.align_signals(norm_a, norm_b, sample_rate=48000)
        self.assertTrue(lag < 0)
        
        aligned_b = norm_b[-lag:]
        min_len = min(len(norm_a), len(aligned_b))
        aligned_a, aligned_b = norm_a[:min_len], aligned_b[:min_len]
        
        spec_a = compare.spec(aligned_a, sample_rate=48000)
        spec_b = compare.spec(aligned_b, sample_rate=48000)
        res = compare.compare(spec_a, spec_b)
        
        # Higher score after alignment/normalization
        self.assertTrue(res.score > 90.0)


if __name__ == "__main__":
    if not HAS_DEPENDENCIES:
        print(f"Warning: Cannot run tests because of missing dependencies: {IMPORT_ERROR_MSG}")
        sys.exit(0)
    unittest.main()
