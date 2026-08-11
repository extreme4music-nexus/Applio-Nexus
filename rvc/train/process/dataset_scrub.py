import os
import librosa
import numpy as np
import soundfile as sf
import torch
import scipy.signal as signal

# Load Silero VAD for intelligent slicing
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
(get_speech_timestamps, _, _, _, _) = utils

def apply_audio_rack(audio, sr, use_gate, gate_db, use_eq, low_cut, high_cut, use_limiter, limiter_db, gain_mode, manual_gain_db):
    """Applies channel strip DSP routines sequentially to raw floating-point audio numpy arrays."""
    # 1. GAIN STAGING
    if gain_mode == "Manual":
        factor = 10 ** (manual_gain_db / 20.0)
        audio = audio * factor
    elif gain_mode == "Auto (RMS Peak)":
        target_rms = 0.15  # Balanced target scaling profile
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)

    # 2. NOISE GATE (Downward Expansion)
    if use_gate:
        threshold = 10 ** (gate_db / 20.0)
        # Smooth windowing tracking to avoid abrupt chatter clicks
        window_size = int(sr * 0.02)
        rms_env = np.sqrt(signal.filtfilt(np.ones(window_size)/window_size, 1, audio**2))
        gate_mask = np.where(rms_env < threshold, 0.0, 1.0)
        # Apply a smooth moving average filter to create an attack/release slope
        smooth_mask = signal.filtfilt(np.ones(window_size)/window_size, 1, gate_mask)
        audio = audio * smooth_mask

    # 3. EQUALIZATION (Butterworth Bandpass Filter)
    if use_eq and low_cut < high_cut:
        nyquist = 0.5 * sr
        low = low_cut / nyquist
        high = high_cut / nyquist
        # Ensure coefficients don't clip outside Nyquist bounds
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        b, a = signal.butter(4, [low, high], btype='band')
        audio = signal.filtfilt(b, a, audio)

    # 4. BRICKWALL LIMITER
    if use_limiter:
        ceiling = 10 ** (limiter_db / 20.0)
        audio = np.clip(audio, -ceiling, ceiling)

    return audio

def scrub_audio(input_path, output_dir, use_gate=False, gate_db=-45.0, use_eq=False, low_cut=80, high_cut=12000, use_limiter=False, limiter_db=-1.0, gain_mode="Auto (RMS Peak)", manual_gain_db=0.0):
    try:
        # Load audio normally to get the source properties
        audio, sr = librosa.load(input_path, sr=None)
        
        # Apply our interactive hardware-style channel rack DSP routines
        audio = apply_audio_rack(audio, sr, use_gate, gate_db, use_eq, low_cut, high_cut, use_limiter, limiter_db, gain_mode, manual_gain_db)

        # Build 16k version explicitly for Silero alignment mapping
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr_16k = 16000

        # --- EXPLICIT ARRAY TYPE COERCION FOR JIT COALESCING ---
        audio_16k = audio_16k.astype(np.float32)
        wav = torch.from_numpy(audio_16k).float()

        # Execute speech segmentation routing with precision type safeguards
        try:
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr_16k)
        except RuntimeError as tensor_err:
            if "expected scalar type Double" in str(tensor_err):
                # Dynamically convert tensor to Float64 format if backend requires it
                wav = wav.double()
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr_16k)
            else:
                raise tensor_err
        # -------------------------------------------------------
        
        os.makedirs(output_dir, exist_ok=True)
        slices_saved = 0
        
        for i, stamp in enumerate(speech_timestamps):
            start_sample = int((stamp['start'] / sr_16k) * sr)
            end_sample = int((stamp['end'] / sr_16k) * sr)
            
            # Check duration: minimum 1.5 seconds clip limit target
            if (end_sample - start_sample) > int(sr * 1.5):
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                slice_path = os.path.join(output_dir, f"{base_name}_slice_{i}.wav")
                
                # Write processed high-quality array segment directly
                sf.write(slice_path, audio[start_sample:end_sample], sr)
                slices_saved += 1
                
        return True, f"Processed {slices_saved} slices successfully with custom DSP Rack configurations applied."
        
    except Exception as e:
        return False, f"Failed execution step: {str(e)}"