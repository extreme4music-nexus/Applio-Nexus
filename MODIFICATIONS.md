# Modifications.md

This document serves as the comprehensive tracking ledger for custom architectural enhancements implemented within this fork of the Applio v3.6.3 ecosystem. These modifications upgrade the standard processing pipeline into a high-performance, hardware-aware, concurrent, and memory-safe parallel chunk orchestration engine.

---

## 1. UI Stabilization, Core Lifecycle Patching & Event Debouncing
* **File Impacted:** `tabs/settings/settings.py`
* **Intent:** Corrects critical application crash loops during hot-reloads, introduces a structural lifecycle protection layer for backend initialization configurations, and eliminates redundant front-end event loop firings.

### Technical Implementation Details
* **Hot-Reload Process Alignment:** Corrected a runtime pointer mapping bug by swapping the internal process interceptor target to reference `os.execl` instead of `os.execv`. This allows automated app restarts to unpack variadic positional script strings (`*args`) natively into independent shell arguments, successfully eliminating process execution `TypeErrors`.
* **Recursive Monkey-Patch Lock:** Introduced a validation state parameter (`_is_patched_by_fork`) to the core initialization sequence of the `Config` class (`RVCConfig`). This check guarantees that custom hardware acceleration hook bindings only initialize once globally, completely preventing infinite nested tracking loops when backend scripts dynamically re-import configuration utilities during rendering tasks.
* **Front-End Notification Debouncer:** Configured disk cache lookups inside the Gradio input event handlers (`change_device_target` and `change_parallel_target`). The system explicitly checks the configuration values stored on disk against fresh incoming changes before processing UI operations, successfully deduplicating identical event firings and permanently eliminating double pop-up notification toasts.

---

## 2. Dynamic Hardware Memory Resource Profiler (OOM Protection Shield)
* **File Impacted:** `rvc/lib/tools/split_audio.py` (`parallel_inference_mapping`)
* **Intent:** Automatically balances high-performance multithreading calculations against available physical system boundaries to guarantee complete background protection against Out-Of-Memory (OOM) fatal crashes.

### Technical Implementation Details
* **CPU Mode Constraint Evaluation:** Queries active system logical thread configurations (`os.cpu_count()`), and dynamically profiles live unallocated system RAM using memory scanning hooks (`psutil.virtual_memory().available`). The pipeline maps a defensive calculation budget weighing roughly ~1.15 GB of host memory per sub-worker process, automatically reserving 2 core threads to keep the web interface and operating system completely responsive under heavy load.
* **GPU Mode Constraint Evaluation:** Directly queries active device VRAM boundaries down to exact byte metrics using native CUDA tracking calls (`torch.cuda.mem_get_info`). The orchestrator allocates a strict resource buffer capacity (~1.25 GB VRAM allocation weight per chunk worker) to dynamically clamp maximum concurrent processing thread counts.
* **Asynchronous Grouped Batch Orchestration:** Instead of sending all split frames down the pipeline simultaneously, the orchestrator divides the compilation queue into isolated, sequential processing sets (execution groups) throttled by the system's dynamic hardware limits. It forces `torch.cuda.empty_cache()` sweeps and aggressive variable collection rules (`gc.collect()`) between groups, allowing long files to be handled smoothly without memory leakage.

---

## 3. Resilient Native Vocal-Chunk Pitch Locking
* **File Impacted:** `rvc/lib/tools/split_audio.py` (`parallel_inference_mapping`)
* **Intent:** Achieves uniform pitch matching across parallel worker loops without encountering missing path errors from volatile external third-party backend packages.

### Technical Implementation Details
* **Active Vocal Chunk Scanner:** When `Proposed Pitch Locking` is checked, the orchestrator intercepts the workflow right after splitting and filters through the segments to isolate the first chunk containing an active vocal amplitude envelope (`np.abs(c).max() > 0.01`), effectively ignoring silent instrumental openings or blank lead-ins.
* **Librosa Probabilistic YIN Integration:** Employs Librosa's probabilistic YIN algorithm (`librosa.pyin`), a standard internal dependency guaranteed to be operational in every local environment, directly on that voice segment. The fundamental tracking window ($f_0$) is tightly constrained to valid human vocal boundaries ($65\text{Hz} - 1000\text{Hz}$) using absolute hop-scaling factors (`hop_length=160`) to fetch clean pitch vectors out-of-the-box.
* **Locked Transposition Injection:** Fetches the true median value of the calculated vocal pitch contour, determines the mathematically precise logarithmic semitone distance against the user-configured `proposed_pitch_threshold`, and locks that shift value uniformly across all sub-workers. It then forces downstream chunk-level local pitch extraction flags to deactivate (`proposed_pitch = False`), maximizing overall processing speeds while keeping a completely cohesive pitch across split borders.

---

## 4. Decoupled Core Inference Integration
* **File Impacted:** `rvc/infer/infer.py` (`convert_audio`)
* **Intent:** Adapts the baseline voice conversion lifecycle to pipe data directly into the background group orchestrator, removing rigid and slow sequential looping mechanisms.

### Technical Implementation Details
* **Decoupled Workflow Processing:** Completely stripped out old, non-optimized processing loops (`for c in chunks:`) from the inner function blocks of the voice converter class.
* **Unified Parameter Bundling:** Configured the pipeline to package all operational user constraints—including custom index matching paths, index blending rates, volume envelopes, and autotune parameters—into a clean properties dictionary (`pipeline_kwargs`).
* **Centralized Orchestration Routing:** Passes the raw unbroken audio tensor, sliced sub-chunks, interval sample boundaries, sampling rates, and configuration properties directly into `parallel_inference_mapping()`. This hands over full structural execution mapping to the advanced multithreaded group engine, yielding faster rendering times and absolute protection against memory crashes.
