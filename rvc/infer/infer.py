import os
import sys
import soxr
import time
import json
import torch
import librosa
import logging
import concurrent.futures
import numpy as np
import soundfile as sf
import noisereduce as nr
from functools import partial
from pedalboard import (
    Pedalboard,
    Chorus,
    Distortion,
    Reverb,
    PitchShift,
    Limiter,
    Gain,
    Bitcrush,
    Clipping,
    Compressor,
    Delay,
)

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.infer.pipeline import Pipeline as VC
from rvc.infer.pipeline import Pipeline
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.lib.tools.split_audio import process_audio, merge_audio, parallel_inference_mapping
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class VoiceConverter:
    """
    An advanced class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method,
    extended to support real-time Multi-Model Morphing Blending and Adaptive Prosody Performance Mapping.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.hubert_model = None  # Initialize the Hubert model (for embedding extraction)
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        
        # Dual-Model Synthesis Support Architecture
        self.net_g = None  # Primary Generator Model A
        self.net_g_B = None  # Secondary Generator Model B
        self.cpt = None  # Primary Checkpoint Model A weights
        self.cpt_B = None  # Secondary Checkpoint Model B weights
        self.loaded_model = None  # Tracked primary weight path
        self.loaded_model_B = None  # Tracked secondary weight path
        
        self.vc = None  # Voice conversion pipeline instance
        self.version = None  # Model A version
        self.version_B = None  # Model B version
        self.n_spk = None  # Number of speakers in primary model
        self.use_f0 = None  # Whether primary model uses F0

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.
        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model = self.hubert_model.to(self.config.device).float()
        self.hubert_model.eval()

    @staticmethod
    def remove_audio_noise(data, sr, reduction_strength=0.7):
        """
        Removes noise from an audio file using the NoiseReduce library.
        """
        try:
            reduced_noise = nr.reduce_noise(
                y=data, sr=sr, prop_decrease=reduction_strength
            )
            return reduced_noise
        except Exception as error:
            print(f"An error occurred removing audio noise: {error}")
            return None

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts an audio file to a specified output format.
        """
        try:
            if output_format != "WAV":
                print(f"Saving audio as {output_format}...")
                audio, sample_rate = librosa.load(input_path, sr=None)
                common_sample_rates = [
                    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000
                ]
                target_sr = min(common_sample_rates, key=lambda x: abs(x - sample_rate))
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
                )
                sf.write(output_path, audio, target_sr, format=output_format.lower())
            return output_path
        except Exception as error:
            print(f"An error occurred converting the audio format: {error}")

    @staticmethod
    def post_process_audio(audio_input, sample_rate, **kwargs):
        board = Pedalboard()
        if kwargs.get("reverb", False):
            reverb = Reverb(
                room_size=kwargs.get("reverb_room_size", 0.5),
                damping=kwargs.get("reverb_damping", 0.5),
                wet_level=kwargs.get("reverb_wet_level", 0.33),
                dry_level=kwargs.get("reverb_dry_level", 0.4),
                width=kwargs.get("reverb_width", 1.0),
                freeze_mode=kwargs.get("reverb_freeze_mode", 0),
            )
            board.append(reverb)
        if kwargs.get("pitch_shift", False):
            pitch_shift = PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0))
            board.append(pitch_shift)
        if kwargs.get("limiter", False):
            limiter = Limiter(
                threshold_db=kwargs.get("limiter_threshold", -6),
                release_ms=kwargs.get("limiter_release", 0.05),
            )
            board.append(limiter)
        if kwargs.get("gain", False):
            gain = Gain(gain_db=kwargs.get("gain_db", 0))
            board.append(gain)
        if kwargs.get("distortion", False):
            distortion = Distortion(drive_db=kwargs.get("distortion_gain", 25))
            board.append(distortion)
        if kwargs.get("chorus", False):
            chorus = Chorus(
                rate_hz=kwargs.get("chorus_rate", 1.0),
                depth=kwargs.get("chorus_depth", 0.25),
                centre_delay_ms=kwargs.get("chorus_delay", 7),
                feedback=kwargs.get("chorus_feedback", 0.0),
                mix=kwargs.get("chorus_mix", 0.5),
            )
            board.append(chorus)
        if kwargs.get("bitcrush", False):
            bitcrush = Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8))
            board.append(bitcrush)
        if kwargs.get("clipping", False):
            clipping = Clipping(threshold_db=kwargs.get("clipping_threshold", 0))
            board.append(clipping)
        if kwargs.get("compressor", False):
            compressor = Compressor(
                threshold_db=kwargs.get("compressor_threshold", 0),
                ratio=kwargs.get("compressor_ratio", 1),
                attack_ms=kwargs.get("compressor_attack", 1.0),
                release_ms=kwargs.get("compressor_release", 100),
            )
            board.append(compressor)
        if kwargs.get("delay", False):
            delay = Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.5),
                feedback=kwargs.get("delay_feedback", 0.0),
                mix=kwargs.get("delay_mix", 0.5),
            )
            board.append(delay)
        return board(audio_input, sample_rate)

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        volume_envelope: float = 1.0,
        protect: float = 0.5,
        hop_length: int = 128,
        split_audio: bool = False,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        embedder_model: str = "contentvec",
        embedder_model_custom: str = None,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        export_format: str = "WAV",
        post_process: bool = False,
        resample_sr: int = 0,
        sid: int = 0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        
        # EXTENSION: Dynamic Blending Matrix & Performance Engine Parameters
        model_path_b: str = None,
        index_path_b: str = None,
        f0_method_b: str = None,  
        performance_grit: float = 0.0,
        performance_breathiness: float = 0.0,
        performance_vibrato_style: str = "None",
        performance_vibrato_intensity: float = 0.0,
        blend_crossover_freq: float = 800.0,
        blend_velocity_switching: bool = False,
        blend_bias: float = 0.5,
        
        # --- NEW ADDITIONS: Feature-Level Neural Morphing Matrix ---
        blend_timbre: float = 0.0,
        blend_prosody: float = 0.0,
        blend_transients: float = 0.0,
        **kwargs,
    ):
        """
        Performs optimized voice conversion on the input audio path, utilizing advanced
        adaptive prosody micro-expression shifts and multi-model blend logic.
        """
        if not model_path:
            print("No baseline model path provided. Aborting conversion.")
            return

        # Setup model pipelines dynamically for unified or multi-model paths
        self.get_vc(model_path, sid, model_path_b=model_path_b)

        start_time = time.time()
        print(f"Converting audio '{audio_input_path}'...")

        audio = load_audio_infer(audio_input_path, 16000, **kwargs)
        audio_max = np.abs(audio).max() / 0.95

        if audio_max > 1:
            audio /= audio_max

        if not self.hubert_model or embedder_model != self.last_embedder_model:
            self.load_hubert(embedder_model, embedder_model_custom)
            self.last_embedder_model = embedder_model

        file_index = (
            index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
            if index_path else ""
        )
        
        file_index_b = (
            index_path_b.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
            if index_path_b else ""
        )

        if self.tgt_sr != resample_sr >= 16000:
            self.tgt_sr = resample_sr

        intervals = None
        if split_audio:
            chunks, intervals = process_audio(audio, 16000)
            print(f"Audio split into {len(chunks)} chunks for parallel rendering.")
        else:
            chunks = [audio]

        final_f0_method_b = f0_method_b if f0_method_b else f0_method

        pipeline_kwargs = {
            "pitch": pitch,
            "f0_method": f0_method,
            "f0_method_b": final_f0_method_b,  
            "file_index": file_index,
            "index_rate": index_rate,
            "pitch_guidance": self.use_f0,
            "volume_envelope": volume_envelope,
            "version": self.version,
            "protect": protect,
            "f0_autotune": f0_autotune,
            "f0_autotune_strength": f0_autotune_strength,
            "proposed_pitch": proposed_pitch,
            "proposed_pitch_threshold": proposed_pitch_threshold,
            
            # Forward performance engine adjustments
            "performance_grit": performance_grit,
            "performance_breathiness": performance_breathiness,
            "performance_vibrato_style": performance_vibrato_style,
            "performance_vibrato_intensity": performance_vibrato_intensity,
            
            # Forward multi-model morph parameters
            "net_g_B": self.net_g_B,
            "file_index_b": file_index_b,
            "version_B": self.version_B,
            "blend_crossover_freq": blend_crossover_freq,
            "blend_velocity_switching": blend_velocity_switching,
            "blend_bias": blend_bias,
            
            "blend_timbre": blend_timbre,
            "blend_prosody": blend_prosody,
            "blend_transients": blend_transients,
        }

        # STRUCTURAL INITIALIZATION SAFETY FALLBACK INJECTION GUARD
        if self.vc is None:
            self.setup_vc_instance()

        # Route conversion vectors directly down to parallel processing sets
        converted_chunks = parallel_inference_mapping(
            inference_worker_func=self.vc.pipeline,
            model=self.hubert_model,
            net_g=self.net_g,
            sid=sid,
            full_audio=audio,
            chunks=chunks,
            intervals=intervals,
            sr=16000,
            split_audio_enabled=split_audio,
            **pipeline_kwargs
        )

        if split_audio:
            audio_opt = merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
        else:
            audio_opt = converted_chunks[0]

        if clean_audio:
            cleaned_audio = self.remove_audio_noise(audio_opt, self.tgt_sr, clean_strength)
            if cleaned_audio is not None:
                audio_opt = cleaned_audio

        if post_process:
            audio_opt = self.post_process_audio(
                audio_input=audio_opt,
                sample_rate=self.tgt_sr,
                **kwargs,
            )

        sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
        output_path_format = audio_output_path.replace(".wav", f".{export_format.lower()}")
        audio_output_path = self.convert_audio_format(audio_output_path, output_path_format, export_format)

        del audio
        del chunks
        del converted_chunks
        if 'audio_opt' in locals():
            del audio_opt
        import gc
        gc.collect()

        elapsed_time = time.time() - start_time
        print(f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds.")

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        **kwargs,
    ):
        """
        Performs voice conversion on a batch of input audio files.
        """
        pid = os.getpid()
        try:
            with open(os.path.join(now_dir, "assets", "infer_pid.txt"), "w") as pid_file:
                pid_file.write(str(pid))
            start_time = time.time()
            print(f"Converting audio batch '{audio_input_paths}'...")
            audio_files = [
                f for f in os.listdir(audio_input_paths)
                if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))
            ]
            print(f"Detected {len(audio_files)} audio files for inference.")
            for a in audio_files:
                new_input = os.path.join(audio_input_paths, a)
                new_output = os.path.join(audio_output_path, os.path.splitext(a)[0] + "_output.wav")
                if os.path.exists(new_output):
                    continue
                self.convert_audio(
                    audio_input_path=new_input,
                    audio_output_path=new_output,
                    **kwargs,
                )
            print(f"Conversion completed at '{audio_input_paths}'.")
            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
        finally:
            if os.path.exists(os.path.join(now_dir, "assets", "infer_pid.txt")):
                os.remove(os.path.join(now_dir, "assets", "infer_pid.txt"))

    def get_vc(self, weight_root, sid, model_path_b=None):
        """
        Loads the primary and optional secondary voice conversion models into memory.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.vc is None or getattr(self, 'net_g', None) is None:
            self.loaded_model = None
            self.loaded_model_B = None

        if not self.loaded_model or self.loaded_model != weight_root:
            self.load_model(weight_root, is_model_b=False)
            
            if self.cpt is not None:
                self.setup_network(is_model_b=False)
                self.loaded_model = weight_root
            else:
                self.loaded_model = None

        if model_path_b:
            if not self.loaded_model_B or self.loaded_model_B != model_path_b:
                self.load_model(model_path_b, is_model_b=True)
                if self.cpt_B is not None:
                    self.setup_network(is_model_b=True)
                    self.loaded_model_B = model_path_b
                else:
                    self.net_g_B = None
                    self.loaded_model_B = None
        else:
            self.net_g_B = None
            self.cpt_B = None
            self.loaded_model_B = None

        if self.cpt is not None or self.net_g is not None:
            self.setup_vc_instance()
        else:
            print("Warning: Context assets not loaded yet. Allocating fallback structure pipeline container.")
            self.setup_vc_instance()

    def cleanup_model(self):
        """
        Cleans up models and explicitly flushes residual memory tracks out of active processors.
        """
        if self.hubert_model is not None:
            del self.net_g, self.net_g_B, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.net_g_B = self.n_spk = self.vc = self.tgt_sr = None
        else:
            if hasattr(self, 'net_g') and self.net_g is not None: del self.net_g
            if hasattr(self, 'net_g_B') and self.net_g_B is not None: del self.net_g_B
            self.net_g = self.net_g_B = None

        if hasattr(self, 'cpt') and self.cpt is not None: del self.cpt
        if hasattr(self, 'cpt_B') and self.cpt_B is not None: del self.cpt_B
        self.cpt = self.cpt_B = None
        self.loaded_model = self.loaded_model_B = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, weight_root, is_model_b=False):
        """
        Loads the target model weights configuration securely from local disk.
        """
        # DIRECT CACHE PATH RESOLUTION FALLBACK TO CAPTURE DROPDOWN STRINGS
        if weight_root and not os.path.isfile(weight_root):
            potential_paths = [
                os.path.join(now_dir, "assets", "weights", weight_root),
                os.path.join(now_dir, "assets", "weights", f"{weight_root}.pth"),
                os.path.join(now_dir, "logs", weight_root, f"{weight_root}.pth"),
                os.path.join(now_dir, "logs", weight_root, weight_root)
            ]
            for p in potential_paths:
                if os.path.isfile(p):
                    weight_root = p
                    break

        if os.path.isfile(weight_root):
            checkpoint = torch.load(weight_root, map_location="cpu", weights_only=True)
            if is_model_b:
                self.cpt_B = checkpoint
            else:
                self.cpt = checkpoint
        else:
            if is_model_b:
                self.cpt_B = None
            else:
                self.cpt = None

    def setup_network(self, is_model_b=False):
        """
        Sets up the generative network architecture configuration based on the target checkpoint.
        """
        target_checkpoint = self.cpt_B if is_model_b else self.cpt
        if target_checkpoint is None:
            return

        tgt_sr = target_checkpoint["config"][-1]
        target_checkpoint["config"][-3] = target_checkpoint["weight"]["emb_g.weight"].shape[0]
        use_f0 = target_checkpoint.get("f0", 1)
        version = target_checkpoint.get("version", "v1")
        text_enc_hidden_dim = 768 if version == "v2" else 256
        vocoder = target_checkpoint.get("vocoder", "HiFi-GAN")

        synthetic_generator = Synthesizer(
            *target_checkpoint["config"],
            use_f0=use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )
        if hasattr(synthetic_generator, "enc_q"):
            del synthetic_generator.enc_q

        synthetic_generator.load_state_dict(target_checkpoint["weight"], strict=False)
        synthetic_generator = synthetic_generator.to(self.config.device).float()
        synthetic_generator.eval()

        if is_model_b:
            self.net_g_B = synthetic_generator
            self.version_B = version
        else:
            self.net_g = synthetic_generator
            self.tgt_sr = tgt_sr
            self.use_f0 = use_f0
            self.version = version

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate.
        """
        if self.tgt_sr is None:
            self.tgt_sr = 48000  # Safe explicit sample rate default fallback
        
        # REMOVED GATED CHECK TO FORCE THE PIPELINE TO INITIALIZE UNCONDITIONALLY
        self.vc = VC(self.tgt_sr, self.config)
        
        if self.cpt is not None:
            self.n_spk = self.cpt["config"][-3]
        elif self.n_spk is None:
            self.n_spk = 1