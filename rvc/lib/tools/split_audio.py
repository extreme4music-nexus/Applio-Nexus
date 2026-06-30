import os
import json
import numpy as np
import librosa
import concurrent.futures
import psutil
import torch
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


def load_saved_device_config():
    """Reads the stored hardware processing device target from disk."""
    now_dir = os.getcwd()
    config_path = os.path.join(now_dir, "assets", "device_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("device", "cpu")
        except Exception:
            pass
    return "cpu"


def parallel_inference_mapping(inference_worker_func, model, net_g, sid, full_audio, chunks, intervals, sr, split_audio_enabled, **kwargs):
    """
    Advanced Background Orchestration Engine. Evaluates the tracking pitch offset on the first 
    active audio chunk using core native librosa utilities, profiles system memory ceilings, 
    and schedules parallel worker arrays inside safe execution group pipelines.
    """
    saved_parallel, saved_lock_pitch, saved_workers = load_saved_parallel_config()
    saved_device = load_saved_device_config().lower()
    
    ui_parallel_enabled = (os.environ.get("APPLIO_PARALLEL", "false").lower() == "true") or saved_parallel
    ui_lock_pitch_enabled = (os.environ.get("APPLIO_PARALLEL_LOCK_PITCH", "false").lower() == "true") or saved_lock_pitch

    # Extract standard baseline parameters safely out of trailing variables
    pitch = kwargs.pop("pitch", 0)
    proposed_pitch_active = kwargs.get("proposed_pitch", False)
    proposed_pitch_threshold = kwargs.get("proposed_pitch_threshold", 155.0)

    # --- FALLBACK PATH: NATIVE SEQUENTIAL EXECUTION LOOP ---
    if not ui_parallel_enabled or not split_audio_enabled:
        print("[Parallel Engine] Parallel optimization disabled or Split Audio unchecked. Running default loop.")
        converted_chunks = []
        try:
            for i, c in enumerate(chunks):
                audio_opt = inference_worker_func(model, net_g, sid, c, pitch, **kwargs)
                converted_chunks.append(audio_opt)
            return converted_chunks
        finally:
            if 'audio_opt' in locals(): del audio_opt

    print(f"[Parallel Engine] Orchestrator online. Total chunks to process: {len(chunks)}")

    # --- NEW OPTIMIZATION: FIRST VOCAL CHUNK PITCH CALCULATION & TRANSPOSITION LOCK ---
    worker_proposed_pitch = proposed_pitch_active

    if ui_lock_pitch_enabled and proposed_pitch_active and len(chunks) > 0:
        print("[Parallel Engine] Force Uniform Pitch Active. Scanning for first active vocal chunk to establish master baseline offset...")
        
        # Locate the first chunk that isn't just silence/noise to ensure accurate pitch analysis
        target_chunk = chunks[0]
        for c in chunks:
            if np.abs(c).max() > 0.01:
                target_chunk = c
                break
                
        try:
            # Use native librosa.pyin (built into all environments) for safe fundamental frequency tracking
            # Restrict frequency search bounds safely to standard human vocal performance ranges (65Hz - 1000Hz)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                target_chunk, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C6'), 
                sr=sr,
                hop_length=160
            )
            
            # Extract valid tracking vectors
            valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
            f0_cohesive = np.median(valid_f0) if len(valid_f0) > 0 else 0.0
            
            if 0 < f0_cohesive < proposed_pitch_threshold:
                # Apply standard logarithmic semitone shift distance formulas
                calculated_offset = int(np.round(12 * np.log2(proposed_pitch_threshold / f0_cohesive)))
                pitch = pitch + calculated_offset
                print(f"[Parallel Engine] First chunk tracking complete. Applied uniform transposition shift: {calculated_offset} semitones.")
            else:
                print("[Parallel Engine] First voice chunk range is already optimal. Maintaining default baseline keys.")
            
            # Disable chunk-level pitch detection for all chunks to maximize speed
            worker_proposed_pitch = False
            kwargs["proposed_pitch"] = False
        except Exception as pyin_err:
            print(f"[Parallel Engine] Warning: Native chunk pitch tracking encountered an error ({pyin_err}). Falling back to chunk-level tracking.")

    # --- THE BACKGROUND HARDWARE PROFILE MANAGER (ANTI-OOM CEILING SCANNER) ---
    logical_cores = os.cpu_count() or 1
    available_sys_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    max_workers = min(len(chunks), logical_cores)
    if saved_workers is not None:
        try:
            max_workers = min(int(saved_workers), logical_cores)
        except ValueError:
            pass

    is_gpu_mode = "cpu" not in saved_device and (torch.cuda.is_available() or "cuda" in saved_device)
    
    if is_gpu_mode:
        try:
            device_idx = 0
            if "cuda:" in saved_device:
                device_idx = int(saved_device.split(":")[-1].split(" ")[0])
            vram_free_gb = torch.cuda.mem_get_info(device_idx)[0] / (1024 ** 3)
            
            # Defensive Footprint Budget Mapping: Allocate ~1.25 GB VRAM ceiling per worker process
            estimated_safe_gpu_workers = int(vram_free_gb // 1.25)
            if estimated_safe_gpu_workers < 1:
                estimated_safe_gpu_workers = 1
                
            max_workers = min(max_workers, estimated_safe_gpu_workers)
            print(f"[Parallel Engine] Target backend: GPU. Available VRAM: {vram_free_gb:.2f} GB. Max safe concurrent workers: {max_workers}")
        except Exception:
            max_workers = min(max_workers, 2)
            print(f"[Parallel Engine] Hardware device query failed. Throttling concurrent worker capacity safely to: {max_workers}")
    else:
        # Defensive Footprint Budget Mapping: Allocate ~1.15 GB system RAM weight per core worker
        estimated_safe_cpu_workers = int(available_sys_ram_gb // 1.15)
        if estimated_safe_cpu_workers < 1:
            estimated_safe_cpu_workers = 1
            
        # Keep 2 core threads open so system/Gradio processes don't freeze or lag
        safe_core_ceiling = max(1, logical_cores - 2)
        max_workers = min(max_workers, estimated_safe_cpu_workers, safe_core_ceiling)
        print(f"[Parallel Engine] Target backend: CPU. Available RAM: {available_sys_ram_gb:.2f} GB. Max safe concurrent workers: {max_workers}")

    # --- ADVANCED BATCHED EXECUTION GROUP ORCHESTRATION ---
    total_chunks = len(chunks)
    converted_chunks_map = {}
    
    # Group chunk arrays cleanly into safe chunks matching our calculated memory limits
    chunk_groups = [chunks[x:x+max_workers] for x in range(0, total_chunks, max_workers)]
    print(f"[Parallel Engine] Divided work into {len(chunk_groups)} execution group blocks to prevent out-of-memory issues.")

    try:
        global_index_tracker = 0
        for group_idx, current_group in enumerate(chunk_groups):
            print(f"[Parallel Engine] Processing Execution Group [{group_idx + 1}/{len(chunk_groups)}] (Size: {len(current_group)} chunks)...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(current_group)) as executor:
                futures = []
                for chunk in current_group:
                    worker_kwargs = kwargs.copy()
                    worker_kwargs["proposed_pitch"] = worker_proposed_pitch
                    
                    futures.append(executor.submit(
                        inference_worker_func,
                        model,
                        net_g,
                        sid,
                        chunk,
                        pitch,
                        **worker_kwargs
                    ))
                
                for future in futures:
                    result_audio = future.result()
                    converted_chunks_map[global_index_tracker] = result_audio
                    global_index_tracker += 1
                    
            if is_gpu_mode:
                torch.cuda.empty_cache()

        # Rebuild final output array sequencing perfectly matching original timeline locations
        final_results = [converted_chunks_map[idx] for idx in range(total_chunks)]
        print("[Parallel Engine] All execution batches completed successfully. Merging chunks.")
        return final_results

    except Exception as e:
        print(f"[Parallel Engine] Critical failure inside asynchronous group worker processing: {e}")
        raise e
    finally:
        if 'futures' in locals(): del futures
        if 'converted_chunks_map' in locals(): del converted_chunks_map
        import gc
        gc.collect()
