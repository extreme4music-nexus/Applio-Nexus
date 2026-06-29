import os
import sys
import json
import gradio as gr
import torch

# Fix reboot bug: clear any inherited CPU-masking from a previous session 
# before torch/CUDA can initialize and cache the device environment state.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Also intercept runtime process replacement (reboot) to ensure clean inheritance
def _clean_cuda_mask_before_restart():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

try:
    _orig_execv = os.execv
    def _patched_execv(executable, args):
        _clean_cuda_mask_before_restart()
        return _orig_execv(executable, args)
    os.execv = _patched_execv
except AttributeError:
    pass

try:
    _orig_execl = os.execl
    def _patched_execl(executable, *args):
        _clean_cuda_mask_before_restart()
        return _orig_execl(executable, *args)
    os.execl = _patched_execl
except AttributeError:
    pass

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

from tabs.settings.sections.presence import presence_tab
from tabs.settings.sections.realtime_audio import realtime_audio_tab
from tabs.settings.sections.themes import theme_tab
from tabs.settings.sections.version import version_tab
from tabs.settings.sections.lang import lang_tab
from tabs.settings.sections.restart import restart_tab
from tabs.settings.sections.model_author import model_author_tab
from tabs.settings.sections.precision import precision_tab
from tabs.settings.sections.filter import filter_tab, get_filter_trigger

# Permanent storage configuration file paths
DEVICE_CONFIG_PATH = os.path.join(now_dir, "assets", "device_config.json")
PARALLEL_CONFIG_PATH = os.path.join(now_dir, "assets", "parallel_config.json")


def load_saved_device():
    """Reads the permanently stored hardware device selection from disk."""
    if os.path.exists(DEVICE_CONFIG_PATH):
        try:
            with open(DEVICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("device", None)
        except Exception:
            pass
    return None


def save_device(device_string):
    """Writes the selected hardware device preference permanently to disk."""
    try:
        os.makedirs(os.path.dirname(DEVICE_CONFIG_PATH), exist_ok=True)
        with open(DEVICE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"device": device_string}, f, indent=4)
    except Exception as e:
        print(f"[Hardware Selector] Error saving device config: {e}")


def apply_device_patch_to_backend(device_string):
    """Forces currently running memory modules and variables to update instantly."""
    clean_device = device_string.split(" ")[0].lower()
    
    # Sync environment flag for external backend tasks
    os.environ["APPLIO_DEVICE"] = clean_device
    
    # --- SUBPROCESS MASK FOR TRAINING BACKEND ---
    if "cpu" in clean_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[Hardware Selector] CPU Mode activated. Masking GPU for all background training tasks.")
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print(f"[Hardware Selector] GPU Acceleration enabled. Active target: {clean_device}")
    
    for mod_name, mod in list(sys.modules.items()):
        if "config" in mod_name or "rvc" in mod_name:
            if hasattr(mod, "config"):
                cfg = getattr(mod, "config")
                if hasattr(cfg, "device"):
                    cfg.device = clean_device
                    if "cpu" in clean_device:
                        cfg.is_half = False
            if hasattr(mod, "Config"):
                cfg_cls = getattr(mod, "Config")
                if hasattr(cfg_cls, "device"):
                    cfg_cls.device = clean_device


def load_saved_parallel():
    """Reads the permanently stored parallelism selection from disk.
    Returns a tuple containing (parallel_bool, lock_pitch_bool)
    """
    if os.path.exists(PARALLEL_CONFIG_PATH):
        try:
            with open(PARALLEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("parallel", False), data.get("lock_pitch", False)
        except Exception:
            pass
    return False, False


def save_parallel(parallel_bool, lock_pitch_bool):
    """Writes the selected parallelism preference permanently to disk."""
    try:
        os.makedirs(os.path.dirname(PARALLEL_CONFIG_PATH), exist_ok=True)
        with open(PARALLEL_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"parallel": parallel_bool, "lock_pitch": lock_pitch_bool}, f, indent=4)
    except Exception as e:
        print(f"[Parallelism Selector] Error saving parallel config: {e}")


def apply_parallel_patch_to_backend(parallel_bool, lock_pitch_bool):
    """Updates environment variables and passes configuration flags."""
    os.environ["APPLIO_PARALLEL"] = str(parallel_bool).lower()
    os.environ["APPLIO_PARALLEL_LOCK_PITCH"] = str(lock_pitch_bool).lower()
    
    # Clear out any hard global environment caps so Applio can handle num_workers natively
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("MKL_NUM_THREADS", None)
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
    os.environ.pop("VECLIB_MAXIMUM_THREADS", None)
    os.environ.pop("NUMEXPR_NUM_THREADS", None)
    
    if not parallel_bool:
        print("[Parallelism Selector] Custom parallelism optimization disabled. Applio default num_workers active.")
    else:
        lock_status = "Enabled" if lock_pitch_bool else "Disabled"
        print(f"[Parallelism Selector] Parallel multi-threading capabilities active. Pitch Locking: {lock_status}")
    
    for mod_name, mod in list(sys.modules.items()):
        if "config" in mod_name or "rvc" in mod_name:
            if hasattr(mod, "config"):
                cfg = getattr(mod, "config")
                if hasattr(cfg, "parallel"):
                    cfg.parallel = parallel_bool
                if hasattr(cfg, "parallel_lock_pitch"):
                    cfg.parallel_lock_pitch = lock_pitch_bool
            if hasattr(mod, "Config"):
                cfg_cls = getattr(mod, "Config")
                if hasattr(cfg_cls, "parallel"):
                    cfg_cls.parallel = parallel_bool
                if hasattr(cfg_cls, "parallel_lock_pitch"):
                    cfg_cls.parallel_lock_pitch = lock_pitch_bool


# --- GLOBAL BACKEND STARTUP HOOK ---
try:
    from rvc.configs.config import Config as RVCConfig
except ImportError:
    try:
        from configs.config import Config as RVCConfig
    except ImportError:
        RVCConfig = None

if RVCConfig is not None:
    try:
        original_init = RVCConfig.__init__
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Load and apply Hardware settings
            saved_device = load_saved_device()
            if saved_device:
                clean_dev = saved_device.split(" ")[0].lower()
                self.device = clean_dev
                if "cpu" in clean_dev:
                    self.is_half = False
                    
            # Load and apply Parallelism settings
            saved_parallel, saved_lock_pitch = load_saved_parallel()
            self.parallel = saved_parallel
            self.parallel_lock_pitch = saved_lock_pitch
            apply_parallel_patch_to_backend(saved_parallel, saved_lock_pitch)
            
        RVCConfig.__init__ = patched_init
        print("[Hardware & Parallelism Selector] Permanent RVC backend hooks bound successfully.")
    except Exception as e:
        print(f"[Hardware & Parallelism Selector] Failed to bind startup hook: {e}")


def get_available_devices():
    """Queries your computer hardware to display matching available devices."""
    device_list = ["CPU"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_list.append(f"CUDA:{i} ({torch.cuda.get_device_name(i)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_list.append("MPS (Apple Silicon)")
    try:
        import torch_directml
        if torch_directml.is_available():
            for i in range(torch_directml.device_count()):
                device_list.append(f"DirectML:{i}")
    except ImportError:
        pass
    return device_list


def change_device_target(selected_device):
    """Fired immediately when the dropdown option changes in the UI."""
    save_device(selected_device)
    apply_device_patch_to_backend(selected_device)
    print(f"[Hardware Selector] Device configuration locked to: {selected_device}")
    gr.Info(f"Device configuration permanently locked to {selected_device}!")


def change_parallel_target(selected_parallel, selected_lock_pitch):
    """Fired immediately when the parallelism checkboxes change."""
    save_parallel(selected_parallel, selected_lock_pitch)
    apply_parallel_patch_to_backend(selected_parallel, selected_lock_pitch)
    print(f"[Parallelism Selector] State updated - Parallel: {selected_parallel}, Lock Pitch: {selected_lock_pitch}")
    gr.Info(f"Parallelism configuration permanently synchronized!")


def settings_tab(filter_state_trigger=None):
    if filter_state_trigger is None:
        filter_state_trigger = get_filter_trigger()

    with gr.TabItem(label=i18n("General")):
        filter_component = filter_tab()

        filter_component.change(
            fn=lambda checked: gr.update(value=str(checked)),
            inputs=[filter_component],
            outputs=[filter_state_trigger],
            show_progress=False,
        )
        presence_tab()
        realtime_audio_tab()
        theme_tab()
        
        # --- Permanent Parallelism Selection Section ---
        gr.Markdown("---")
        with gr.Column():
            gr.Markdown(f"### {i18n('Parallel Processing Settings')}")
            
            initial_parallel, initial_lock_pitch = load_saved_parallel()
            apply_parallel_patch_to_backend(initial_parallel, initial_lock_pitch)
            
            parallel_selector = gr.Checkbox(
                value=initial_parallel,
                label=i18n("Enable Parallel processing optimization (Multithreading / Batch parallelism)"),
                interactive=True
            )
            
            lock_pitch_selector = gr.Checkbox(
                value=initial_lock_pitch,
                label=i18n("Force uniform pitch calculation (Locks automated pitch matching across parallel workers)"),
                interactive=True
            )
            
            parallel_selector.change(
                fn=change_parallel_target,
                inputs=[parallel_selector, lock_pitch_selector],
                outputs=[],
            )

            lock_pitch_selector.change(
                fn=change_parallel_target,
                inputs=[parallel_selector, lock_pitch_selector],
                outputs=[],
            )
            
        # --- Permanent Hardware Selection Section (Moved to General Menu) ---
        gr.Markdown("---")
        with gr.Column():
            gr.Markdown(f"### {i18n('Hardware Acceleration Settings')}")
            
            system_devices = get_available_devices()
            
            # Read previously saved choice on layout generation, otherwise default to CUDA/CPU
            saved_preference = load_saved_device()
            if saved_preference in system_devices:
                initial_selection = saved_preference
                apply_device_patch_to_backend(saved_preference)
            else:
                initial_selection = "CUDA:0" if torch.cuda.is_available() else "CPU"
                apply_device_patch_to_backend(initial_selection)
            
            hardware_selector = gr.Dropdown(
                choices=system_devices,
                value=initial_selection,
                label=i18n("Select Processing Device (CPU / iGPU / GPU)"),
                interactive=True
            )
            
            hardware_selector.change(
                fn=change_device_target,
                inputs=[hardware_selector],
                outputs=[],
            )
        gr.Markdown("---")
        
        version_tab()
        lang_tab()
        restart_tab()
        
    with gr.TabItem(label=i18n("Training")):
        model_author_tab()
        precision_tab()
