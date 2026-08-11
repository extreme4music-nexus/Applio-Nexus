import os
import gc
import sys
import torch
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import numpy as np
from scipy import signal
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)


class AudioProcessor:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """

    @staticmethod
    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.
        """
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        adjusted_audio = (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
        return adjusted_audio


class Autotune:
    """
    A class for applying autotune to a given fundamental frequency (F0) contour.
    """

    def __init__(self):
        """
        Initializes the Autotune class with a set of reference frequencies.
        """
        self.note_dict = [
            49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41,
            87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59,
            146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08,
            246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00,
            415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25,
            698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50,
        ]

    def autotune_f0(self, f0, f0_autotune_strength):
        """
        Autotunes a given F0 contour by snapping each frequency to the closest reference frequency.
        """
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            if freq <= 0:
                autotuned_f0[i] = 0
                continue
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0


class Pipeline:
    """
    The extended pipeline class for performing voice conversion, including preprocessing, 
    Performance Engine expression adjustments, Multi-Model Blending node processing, and final synthesis.
    """

    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.autotune = Autotune()

    def get_f0(
        self,
        x,
        p_len,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        performance_vibrato_style: str = "None",
        performance_vibrato_intensity: float = 0.0,
    ):
        """
        Estimates the fundamental frequency (F0) of an audio signal, containing custom 
        modulations for adaptive style vibrato emulation curves.
        """
        if f0_method == "crepe":
            model = CREPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "full")
            del model
        elif f0_method == "crepe-tiny":
            model = CREPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, self.f0_min, self.f0_max, p_len, "tiny")
            del model
        elif f0_method == "rmvpe":
            model = RMVPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, filter_radius=0.03)
            del model
        elif f0_method == "fcpe":
            model = FCPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, p_len, filter_radius=0.006)
            del model

        # Apply chromatic snapping autotune limits if active
        if f0_autotune:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch:
            limit = 12
            valid_f0 = np.where(f0 > 0)[0]
            if len(valid_f0) < 2:
                up_key = 0
            else:
                median_f0 = float(np.median(np.interp(np.arange(len(f0)), valid_f0, f0[valid_f0])))
                if median_f0 <= 0 or np.isnan(median_f0):
                    up_key = 0
                else:
                    up_key = max(-limit, min(limit, int(np.round(12 * np.log2(proposed_pitch_threshold / median_f0)))))
            print("calculated pitch offset:", up_key)
            f0 *= pow(2, (pitch + up_key) / 12)
        else:
            f0 *= pow(2, pitch / 12)

        # --- PERFORMANCE ENGINE: DYNAMIC ADAPTIVE VIBRATO EMULATION LAYER ---
        if performance_vibrato_style != "None" and performance_vibrato_intensity > 0.0:
            style_freqs = {"Pop": 5.5, "Jazz": 4.5, "Opera": 6.5}
            target_lfo_freq = style_freqs.get(performance_vibrato_style, 5.5)
            
            # Frame allocation properties (10ms step sizes)
            total_frames = len(f0)
            time_axis = np.arange(total_frames) * (self.time_step / 1000.0)
            
            # Formulate baseline LFO phase structure featuring realistic micro-expression jitter noise
            jitter = np.random.normal(0, 0.08, size=total_frames)
            lfo_wave = np.sin(2 * np.pi * target_lfo_freq * time_axis + jitter)
            
            # Scan voiced contours using an active running window to identify sustained note sections
            lookahead_frames = 20  # 200ms evaluation bounds
            vibrato_envelope = np.zeros(total_frames)
            
            for f_idx in range(total_frames):
                if f0[f_idx] > 0:
                    start_b = max(0, f_idx - lookahead_frames)
                    end_b = min(total_frames, f_idx + lookahead_frames)
                    voiced_slice = f0[start_b:end_b]
                    
                    # FIX: Define the filter variable BEFORE using it in the slice condition
                    # We create the mask based on voiced_slice itself
                    voiced_pts = voiced_slice[voiced_slice > 0]
                    
                    if len(voiced_pts) > 12 and np.std(voiced_pts) < 15.0:
                        vibrato_envelope[f_idx] = 1.0  # Note is held constant, trigger vibrato envelope
            
            # Apply smooth box filters to soften modulation transitions
            vibrato_envelope = np.convolve(vibrato_envelope, np.ones(10)/10.0, mode='same')
            max_semitone_variance = performance_vibrato_intensity * 1.15
            
            # Inject calculated frequency shifts back into the processing array
            pitch_shift_factors = np.power(2, (lfo_wave * max_semitone_variance * vibrato_envelope) / 12.0)
            f0[f0 > 0] *= pitch_shift_factors[f0 > 0]

        # Quantize tracking arrays to 255 buckets for model embedding requirements
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        performance_grit: float = 0.0,
        performance_breathiness: float = 0.0,
        **kwargs,
    ):
        """
        Performs structural sound synthesis on an isolated audio segment chunk.
        """
        e_model = kwargs.get("embedder_model", "contentvec")
        with torch.no_grad():
            pitch_guidance = pitch is not None and pitchf is not None
            feats_raw = torch.from_numpy(audio0).float()
            feats_raw = feats_raw.mean(-1) if feats_raw.dim() == 2 else feats_raw
            assert feats_raw.dim() == 1, feats_raw.dim()
            feats_raw = feats_raw.view(1, -1).to(self.device)
            
            # Extract speaker layers using baseline hidden configurations
            feats = model(feats_raw)["last_hidden_state"]
            feats_A = model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            feats0 = feats_A.clone() if pitch_guidance else None
            
            if index is not None and big_npy is not None:
                feats_A = self._retrieve_speaker_embeddings(feats_A, index, big_npy, index_rate)
            
            # --- NEW ADDITION: FEATURE-LEVEL NEURAL MORPH MATRIX INJECTION ---
            blend_net_g_B = kwargs.get("blend_net_g_B", None)
            blend_timbre = kwargs.get("blend_timbre", 0.0)
            blend_transients = kwargs.get("blend_transients", 0.0)
            
            if blend_net_g_B is not None and (blend_timbre > 0.0 or blend_transients > 0.0):
                blend_index_B = kwargs.get("blend_index_B", None)
                blend_big_npy_B = kwargs.get("blend_big_npy_B", None)
                version_B = kwargs.get("version_B", "v1")
                
                feats_B = model.final_proj(feats[0]).unsqueeze(0) if version_B == "v1" else feats
                if blend_index_B is not None and blend_big_npy_B is not None:
                    feats_B = self._retrieve_speaker_embeddings(feats_B, blend_index_B, blend_big_npy_B, index_rate)
                
                # Dynamic Linear Dimension Adapter (Prevents V1 vs V2 Shape Crashes)
                dim_A = feats_A.shape[-1]
                dim_B = feats_B.shape[-1]
                if dim_A != dim_B:
                    if dim_A == 256 and dim_B == 768:
                        proj = torch.nn.Linear(256, 768).to(self.device).to(feats_A.dtype)
                        feats_A = proj(feats_A)
                    elif dim_A == 768 and dim_B == 256:
                        proj = torch.nn.Linear(256, 768).to(self.device).to(feats_B.dtype)
                        feats_B = proj(feats_B)
                        
                # 3. Content-Space Normalization (Crucial for multi-embedder setups)
                # Different embedders have different mean/variance.
                # We Z-score normalize both before blending to prevent gain spikes.
                feats_A = (feats_A - feats_A.mean()) / (feats_A.std() + 1e-6)
                feats_B = (feats_B - feats_B.mean()) / (feats_B.std() + 1e-6)                
                
                # 1. Morph Timbre (Vocal Color)
                feats_morphed = torch.lerp(feats_A, feats_B, blend_timbre)
                
                # 2. Morph Transients (Voiceless Consonants)
                if pitch_guidance and blend_transients != blend_timbre:
                    p_len_temp = min(audio0.shape[0] // self.window, feats_morphed.shape[1])
                    pitchf_np = pitchf[:, :p_len_temp].cpu().numpy()[0]
                    for frame_idx in range(min(feats_morphed.shape[1], len(pitchf_np))):
                        if pitchf_np[frame_idx] == 0:
                            feats_morphed[:, frame_idx, :] = torch.lerp(
                                feats_A[:, frame_idx, :], 
                                feats_B[:, frame_idx, :], 
                                blend_transients
                            )
                feats_A = feats_morphed
            
            # Lock the extracted/morphed features back into the processing track
            feats = feats_A
            # ------------------------------------------------------------------

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len].float()
                
                # --- PERFORMANCE ENGINE: PITCH-SYNCHRONIZED GRIT & GROWL MATRIX ---
                if performance_grit > 0.0:
                    # Apply a sub-harmonic saturation function to voiced segments to introduce gravelly textures
                    growl_wave = torch.sin(pitchf * 0.5) * performance_grit * 14.5
                    pitchf = torch.where(pitchf > 0, pitchf + growl_wave, pitchf)
                
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (1 - pitchff.unsqueeze(-1))
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None

            # --- PERFORMANCE ENGINE: LATENT BREATHINESS NOISE COUPLER ---
            if performance_breathiness > 0.0:
                # Shape a Gaussian white noise profile directly into the latent representations
                breath_noise = torch.randn_like(feats) * (performance_breathiness * 0.12)
                feats = feats + breath_noise

            p_len_tensor = torch.tensor([p_len], device=self.device).long()
            audio1 = (net_g.infer(feats.float(), p_len_tensor, pitch, pitchf, sid)[0][0, 0]).data.cpu().float().numpy()
            
            del feats, feats0, p_len_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return audio1

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)
        return feats

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        f0_autotune,
        f0_autotune_strength,
        proposed_pitch,
        proposed_pitch_threshold,
        **kwargs,
    ):
        """
        The main processing pipeline function, managing multi-model blending crossovers 
        and velocity matrix assignments.
        """
        # Read performance parameters and optional secondary model paths safely out of keyword tracking arguments
        p_grit = kwargs.get("performance_grit", 0.0)
        p_breath = kwargs.get("performance_breathiness", 0.0)
        p_vibrato_style = kwargs.get("performance_vibrato_style", "None")
        p_vibrato_intensity = kwargs.get("performance_vibrato_intensity", 0.0)
        
        net_g_B = kwargs.get("net_g_B", None)
        file_index_b = kwargs.get("file_index_b", "")
        version_B = kwargs.get("version_B", "v1")
        blend_crossover_freq = kwargs.get("blend_crossover_freq", 800.0)
        blend_velocity_switching = kwargs.get("blend_velocity_switching", False)
        blend_bias = kwargs.get("blend_bias", 0.5)
        e_model = kwargs.get("embedder_model", "contentvec")
        
        # --- NEW ADDITION: Extract morphing properties safely ---
        blend_timbre = kwargs.get("blend_timbre", 0.0)
        blend_prosody = kwargs.get("blend_prosody", 0.0)
        blend_transients = kwargs.get("blend_transients", 0.0)
        f0_method_b = kwargs.get("f0_method_b", f0_method)

        # Build search blocks for Model B if available
        index_B, big_npy_B = None, None
        if net_g_B is not None and file_index_b != "" and os.path.exists(file_index_b) and index_rate > 0:
            try:
                index_B = faiss.read_index(file_index_b)
                big_npy_B = index_B.reconstruct_n(0, index_B.ntotal)
            except Exception as e:
                print(f"An error occurred reading index file B: {e}")
                index_B = big_npy_B = None

        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"An error occurred reading the FAISS index: {error}")
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        
        if pitch_guidance:
            pitch_coarse, pitch_freq = self.get_f0(
                audio_pad, p_len, f0_method, pitch, f0_autotune, f0_autotune_strength,
                proposed_pitch, proposed_pitch_threshold, p_vibrato_style, p_vibrato_intensity
            )
            
            # --- NEW ADDITION: PROSODY (PITCH) NEURAL MORPHING ---
            if net_g_B is not None and blend_prosody > 0.0:
                pitch_coarse_B, pitch_freq_B = self.get_f0(
                    audio_pad, p_len, f0_method_b, pitch, f0_autotune, f0_autotune_strength,
                    proposed_pitch, proposed_pitch_threshold, p_vibrato_style, p_vibrato_intensity
                )
                
                # Interpolate pitch sequences dynamically
                voiced_mask = (pitch_freq > 0) & (pitch_freq_B > 0)
                pitch_freq[voiced_mask] = (1.0 - blend_prosody) * pitch_freq[voiced_mask] + blend_prosody * pitch_freq_B[voiced_mask]
                
                # Fallback boundaries for trailing mismatches
                u_mask_B = (pitch_freq_B > 0) & (pitch_freq <= 0)
                pitch_freq[u_mask_B] = pitch_freq_B[u_mask_B]
                pitch_coarse = np.rint((1.0 - blend_prosody) * pitch_coarse + blend_prosody * pitch_coarse_B).astype(int)
            # -----------------------------------------------------

            pitch_coarse = pitch_coarse[:p_len]
            pitch_freq = pitch_freq[:p_len]
            if self.device == "mps":
                pitch_freq = pitch_freq.astype(np.float32)
            pitch_coarse = torch.tensor(pitch_coarse, device=self.device).unsqueeze(0).long()
            pitch_freq = torch.tensor(pitch_freq, device=self.device).unsqueeze(0).float()

        # Iterate through audio chunk slices and compile arrays through selected generators
        # Iterate through audio chunk slices
        # Iterate through audio chunk slices and compile arrays through selected generators
        for t in opt_ts:
            t = t // self.window * self.window
            chunk_audio = audio_pad[s : t + self.t_pad2 + self.window]
            
            p_coarse_slice = pitch_coarse[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None
            p_freq_slice = pitch_freq[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None
            
            # Synthesize output via baseline Model A (Feature Morphing handles both models internally inside voice_conversion)
            # This generates ONE perfectly phase-aligned, synchronized chunk natively!
            out_chunk = self.voice_conversion(
                model, net_g, sid_tensor, chunk_audio, p_coarse_slice, p_freq_slice,
                index, big_npy, index_rate, version, protect, p_grit, p_breath,
                blend_net_g_B=net_g_B, blend_index_B=index_B, blend_big_npy_B=big_npy_B,
                version_B=version_B, blend_timbre=blend_timbre, blend_transients=blend_transients,
                f0_method_b=f0_method_b
            )[self.t_pad_tgt : -self.t_pad_tgt]
            
            audio_opt.append(out_chunk)
            s = t

        # Handle final trailing frame slices
        chunk_audio_last = audio_pad[s:]
        p_coarse_last = pitch_coarse[:, s // self.window :] if (pitch_guidance and t is not None) else (pitch_coarse if pitch_guidance else None)
        p_freq_last = pitch_freq[:, s // self.window :] if (pitch_guidance and t is not None) else (pitch_freq if pitch_guidance else None)
        
        out_last = self.voice_conversion(
            model, net_g, sid_tensor, chunk_audio_last, p_coarse_last, p_freq_last,
            index, big_npy, index_rate, version, protect, p_grit, p_breath,
            blend_net_g_B=net_g_B, blend_index_B=index_B, blend_big_npy_B=big_npy_B,
            version_B=version_B, blend_timbre=blend_timbre, blend_transients=blend_transients,
            f0_method_b=f0_method_b
        )[self.t_pad_tgt : -self.t_pad_tgt]
        
        audio_opt.append(out_last)

        # --- UNIFIED RE-SYNCHRONIZATION AND POST-PROCESSING ---
        audio_opt = np.concatenate(audio_opt)
        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(audio, self.sample_rate, audio_opt, self.tgt_sr, volume_envelope)
        
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
            
        if pitch_guidance:
            del pitch_coarse, pitch_freq
        del sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt

    def _blend_audio_outputs(self, out_A, out_B, raw_source, crossover_f, velocity_sw, static_bias):
        """
        Calculates frequency-split crossovers and dynamic envelope matrix weighting 
        to merge output tracks.
        """
        # Safeguard identical dimensions across calculation paths
        min_len = min(len(out_A), len(out_B))
        out_A, out_B = out_A[:min_len], out_B[:min_len]
        
        # Calculate localized blending bias based on velocity switches if active
        if velocity_sw:
            # Analyze raw input track dynamics using a rolling window
            source_rms = librosa.feature.rms(y=raw_source, frame_length=512, hop_length=256)[0]
            if len(source_rms) == 0:
                calculated_bias = static_bias
            else:
                normalized_rms = (source_rms - source_rms.min()) / (source_rms.max() - source_rms.min() + 1e-5)
                mean_rms_weight = float(np.mean(normalized_rms))
                # Dynamic crossfade mapping: low volume favors Model A, high volume favors Model B
                calculated_bias = np.clip(mean_rms_weight, 0.0, 1.0)
        else:
            calculated_bias = static_bias

        # Formulate Butterworth crossover filter parameters matching output sampling configurations
        nyquist = self.tgt_sr * 0.5
        clamped_crossover = np.clip(crossover_f, 100.0, nyquist - 100.0)
        Wn = clamped_crossover / nyquist
        
        b_low, a_low = signal.butter(4, Wn, btype='low')
        b_high, a_high = signal.butter(4, Wn, btype='high')
        
        # Route Model A to handle low frequencies and Model B to handle high frequencies/air
        low_frequencies = signal.filtfilt(b_low, a_low, out_A)
        high_frequencies = signal.filtfilt(b_high, a_high, out_B)
        
        # Sum crossover paths while applying the dynamic blending matrix bias
        blended_frequency_track = (low_frequencies * (1.0 - calculated_bias)) + (high_frequencies * calculated_bias)
        return blended_frequency_track