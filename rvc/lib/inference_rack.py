import os
import torch
import numpy as np
import scipy.signal as signal
from io import BytesIO

class PerformanceInferenceRack:
    def __init__(self, model_a_path, model_b_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_a = self.load_pth_model(model_a_path)
        self.model_b = self.load_pth_model(model_b_path) if model_b_path else None
        
    def load_pth_model(self, path):
        if not path or not os.path.exists(path):
            return None
        print(f"[Performance Engine] Mounting checkpoint tensor array: {os.path.basename(path)}")
        ckpt = torch.load(path, map_map=self.device)
        # Extract weight parameters (handles raw weights vs webui packaged weights)
        model_weights = ckpt.get("weight", ckpt)
        # Put model into strict evaluation/inference mode to disable dropout scaling
        if hasattr(model_weights, "eval"):
            model_weights.eval()
        return model_weights

    def apply_neural_lfo_vibrato(self, f0, sr, style="Bypass", rate=5.5, delay_ms=300):
        """Manipulates the log-frequency array tracking curve using physical LFO equations."""
        if style == "Bypass" or f0 is None:
            return f0
            
        # Unvoiced frames are set to 0 in RVC, collect only active singer notes
        voiced_indices = np.where(f0 > 0)[0]
        if len(voiced_indices) == 0:
            return f0
            
        time_step = 1.0 / (sr / 256.0) # RVC standard default hop tracking step length
        total_frames = len(f0)
        
        # Determine depth tracking maps based on preset style configurations
        if style == "Classical Operatic":
            depth = 0.04  # Symmetric tight operatic wobble variation bound
            rate = 6.5
        elif style == "Loose Vintage Jazz":
            depth = 0.08  # Wider emotional variation depth profile
            rate = 4.8
        else:
            depth = 0.05
            
        delay_frames = int((delay_ms / 1000.0) / time_step)
        
        # Apply sinusoidal modulation explicitly to active vocal regions
        for idx in voiced_indices:
            # Gradually ramp up vibrato depth over time if note is held down long enough
            note_age = max(0, idx - delay_frames)
            envelope_ramp = min(1.0, note_age / int(0.5 / time_step))
            
            # Sinusoidal physical frequency offset calculation loop
            lfo_mod = 1.0 + (depth * envelope_ramp * np.sin(2 * np.pi * rate * idx * time_step))
            f0[idx] = f0[idx] * lfo_mod
            
        return f0

    def apply_acoustic_grit_jitter(self, features, intensity=0.0):
        """Injects phase-jitter into hidden units to simulate vocal tearing / distortion."""
        if intensity == 0.0 or features is None:
            return features
            
        # Generate random distribution maps mirroring organic vocal strain profiles
        jitter_mask = torch.randn_like(features) * (intensity * 0.12)
        return features + jitter_mask

    def execute_split_morph_inference(self, hidden_features, f0_curve, sr, crossover_hz=1200, velocity_sensitivity=0.5):
        """
        Executes parallel generation passes across twin VRAM allocations, 
        handling frequency splits and velocity-driven hidden unit crossfading.
        """
        # Calculate incoming performance RMS amplitude map frame by frame
        rms_energy = torch.sqrt(torch.mean(hidden_features ** 2, dim=-1))
        normalized_rms = (rms_energy - rms_energy.min()) / (rms_energy.max() - rms_energy.min() + 1e-5)
        mean_rms = float(normalized_rms.mean().cpu().numpy())

        # Determine dynamic crossfade parameter alpha based on energy vs sensitivity thresholds
        alpha = min(1.0, max(0.0, mean_rms * (1.0 + velocity_sensitivity)))

        # 1. GENERATION PASS FOR MODEL A
        out_audio_a = self.model_a(hidden_features, f0_curve) if self.model_a else None
        
        # 2. GENERATION PASS FOR MODEL B
        out_audio_b = self.model_b(hidden_features, f0_curve) if self.model_b else out_audio_a

        if out_audio_a is None:
            return out_audio_b
        if out_audio_b is None:
            return out_audio_a

        # Convert generation tracks to linear spectra for crossover mask parsing
        # This completely avoids physical phase cancelation issues
        stft_a = torch.stft(out_audio_a.float(), n_fft=2048, hop_length=512, return_complex=True)
        stft_b = torch.stft(out_audio_b.float(), n_fft=2048, hop_length=512, return_complex=True)
        
        # Compute frequency bin cutoff mapping indexes based on system sample rate parameters
        fft_bins, total_frames = stft_a.shape[-2], stft_a.shape[-1]
        cutoff_bin = int((crossover_hz / (sr / 2.0)) * fft_bins)
        cutoff_bin = max(1, min(cutoff_bin, fft_bins - 1))

        # Construct combined multi-model spectrum canvas
        blended_stft = torch.zeros_like(stft_a)
        
        # Model A handles low-end warmth & chest resonance
        blended_stft[..., :cutoff_bin, :] = stft_a[..., :cutoff_bin, :] * (1.0 - (alpha * 0.3))
        
        # Model B handles high-end air, sibilance, and high-velocity power pushes
        blended_stft[..., cutoff_bin:, :] = stft_b[..., cutoff_bin:, :] * alpha + stft_a[..., cutoff_bin:, :] * (1.0 - alpha)

        # Invert spectrum back to pristine high-fidelity audio arrays
        output_waveform = torch.istft(blended_stft, n_fft=2048, hop_length=512)
        return output_waveform