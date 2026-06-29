import os
import json
import numpy as np
import librosa
import concurrent.futures
from functools import partial

def process_audio(audio, sr=16000, silence_thresh=-60, min_silence_len=250):
    """
    Splits an audio signal into segments using a fixed frame size and hop size.
    """
    frame_length = int(min_silence_len / 1000 * sr)
    hop_length = frame_length // 2
    intervals = librosa.effects.split(
        audio, top_db=-silence_thresh, frame_length=frame_length, hop_length=hop_length
    )
    audio_segments = [audio[start:end] for start, end in intervals]

    return audio_segments, intervals


def merge_audio(audio_segments_org, audio_segments_new, intervals, sr_orig, sr_new):
    """
    Merges audio segments back into a single audio signal, filling gaps with silence.
    """
    merged_audio = np.array([], dtype=audio_segments_new[0].dtype)
    sr_ratio = sr_new / sr_orig

    for i, (start, end) in enumerate(intervals):
        start_new = int(start * sr_ratio)
        end_new = int(end * sr_ratio)

        original_duration = len(audio_segments_org[i]) / sr_orig
        new_duration = len(audio_segments_new[i]) / sr_new
        duration_diff = new_duration - original_duration

        silence_samples = int(abs(duration_diff) * sr_new)
        silence_compensation = np.zeros(
            silence_samples, dtype=audio_segments_new[0].dtype
        )

        if i == 0 and start_new > 0:
            initial_silence = np.zeros(start_new, dtype=audio_segments_new[0].dtype)
            merged_audio = np.concatenate((merged_audio, initial_silence))

        if duration_diff > 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))

        merged_audio = np.concatenate((merged_audio, audio_segments_new[i]))

        if duration_diff < 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))

        if i < len(intervals) - 1:
            next_start_new = int(intervals[i + 1][0] * sr_ratio)
            silence_duration = next_start_new - end_new
            if silence_duration > 0:
                silence = np.zeros(silence_duration, dtype=audio_segments_new[0].dtype)
                merged_audio = np.concatenate((merged_audio, silence))

    return merged_audio


def load_saved_parallel_config():
    """Reads the stored parallel tab configurations directly from disk."""
    now_dir = os.getcwd()
    config_path = os.path.join(now_dir, "assets", "parallel_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("parallel", False), data.get("lock_pitch", False), data.get("num_workers", None)
        except Exception:
            pass
    return False, False, None


def parallel_inference_mapping(inference_worker_func, full_audio, chunks, intervals, sr, split_audio_enabled, **kwargs):
    """
    Orchestrates execution workflow. Pre-calculates global pitch arrays over the 
    entire un-split audio if pitch locking is active, slices both arrays, 
    and handles true process parallelism safely.
    """
    saved_parallel, saved_lock_pitch, saved_workers = load_saved_parallel_config()
    
    ui_parallel_enabled = (os.environ.get("APPLIO_PARALLEL", "false").lower() == "true") or saved_parallel
    ui_lock_pitch_enabled = (os.environ.get("APPLIO_PARALLEL_LOCK_PITCH", "false").lower() == "true") or saved_lock_pitch

    # --- FALLBACK PATH: SEQUENTIAL EXECUTION LOOP ---
    if not ui_parallel_enabled or not split_audio_enabled:
        print("[Parallel Engine] Parallelism off or Split Audio unselected. Falling back to default mode.")
        converted_chunks = []
        try:
            for i, c in enumerate(chunks):
                audio_opt = inference_worker_func(audio=c, **kwargs)
                converted_chunks.append(audio_opt)
            return converted_chunks
        finally:
            if 'audio_opt' in locals(): 
                del audio_opt

    # --- TRUE PARALLELISM PATH: GLOBAL PITCH PRE-CALCULATION ENGINE ---
    total_chunks = len(chunks)
    max_workers = min(total_chunks, 12)
    if saved_workers is not None:
        try:
            max_workers = int(saved_workers)
        except ValueError:
            pass
    
    lock_status = "ACTIVE" if ui_lock_pitch_enabled else "INACTIVE"
    print(f"[Parallel Engine] Optimization active. Chunks: {total_chunks}, Workers: {max_workers}, Global Pitch Lock: {lock_status}")
    
    sliced_pitch_chunks = [None] * total_chunks

    if ui_lock_pitch_enabled and kwargs.get("proposed_pitch", False):
        print("[Parallel Engine] Performing global pitch calculations on full audio stream before chunk allocation...")
        
        # Extract user threshold constraints safely from passed kwargs
        pitch_method = kwargs.get("f0method", "pm")
        pitch_th = kwargs.get("proposed_pitch_threshold", 0.0) # Read user set pitch threshold
        hop_length = 160  # Default hop length used across standard RVC pipelines
        
        # Dynamically import the native pitch extraction tool from Applio's backend
        try:
            from rvc.lib.predictors.F0Predictor import get_f0_predictor
            # Instantiate the chosen pitch algorithm handler
            predictor = get_f0_predictor(pitch_method, hop_length=hop_length, sampling_rate=sr)
            
            # Compute the global f0 pitch array across the complete, unbroken audio stream
            global_f0, _ = predictor.compute_f0(full_audio, pitch_th)
            
            # Convert intervals from audio sample indices to matching f0 feature frame indices
            # Audio index maps to frame index via: frame = sample / hop_length
            for i, (start_sample, end_sample) in enumerate(intervals):
                start_frame = int(start_sample / hop_length)
                end_frame = int(end_sample / hop_length)
                # Slice the pre-calculated global pitch array cleanly with absolute context preservation
                sliced_pitch_chunks[i] = global_f0[start_frame:end_frame]
                
            print("[Parallel Engine] Global pitch envelope computed and sliced successfully across chunk targets.")
        except Exception as pitch_err:
            print(f"[Parallel Engine] Warning: Failed to compute global pitch pre-calculation ({pitch_err}). Falling back to local chunk calculation.")

    # --- CONCURRENT WORKER DISPATCH LOOP ---
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                # Copy global keyword parameters for each distinct worker
                worker_kwargs = kwargs.copy()
                
                # If global pitch array exists for this chunk, inject it directly and disable local recalculation
                if sliced_pitch_chunks[i] is not None:
                    worker_kwargs["f0_precalculated"] = sliced_pitch_chunks[i]
                    worker_kwargs["proposed_pitch"] = False  # Tells worker loop to bypass extraction step

                # Queue the worker thread execution target
                futures.append(executor.submit(inference_worker_func, audio=chunk, **worker_kwargs))
            
            # Wait and gather execution responses sequentially to ensure audio array ordering matches
            results = [f.result() for f in futures]
        return results
        
    except Exception as e:
        print(f"[Parallel Engine] Critical failure inside thread mapping execution queue: {e}")
        raise e
    finally:
        # Aggressive memory cleanup behaviors to prevent thread-bound OOM conditions
        if 'futures' in locals(): del futures
        if 'sliced_pitch_chunks' in locals(): del sliced_pitch_chunks
        import gc
        gc.collect()
