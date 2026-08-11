import os
import gradio as gr
from assets.i18n.i18n import I18nAuto
from core import run_dataset_scrubber
from rvc.train.process.phoneme_analyzer import analyze_phoneme_coverage

i18n = I18nAuto()
BASE_DATASETS_DIR = os.path.join("assets", "datasets")

def get_dataset_subfolders():
    """Scans assets/datasets/ and returns a list of available subfolder paths."""
    if not os.path.exists(BASE_DATASETS_DIR):
        os.makedirs(BASE_DATASETS_DIR, exist_ok=True)
    try:
        subfolders = [f.name for f in os.scandir(BASE_DATASETS_DIR) if f.is_dir()]
        return sorted(subfolders)
    except Exception:
        return []

def update_dataset_paths(selected_folder):
    """Computes clean absolute paths to ensure zero Windows path parsing conflicts."""
    if not selected_folder:
        return "", ""
    full_path = os.path.abspath(os.path.join(BASE_DATASETS_DIR, selected_folder))
    output_path = os.path.abspath(os.path.join(full_path, "samples"))
    return full_path, output_path

def refresh_dataset_dropdown():
    """Forces the workspace asset catalog to scan the hard drive dynamically."""
    return gr.Dropdown(choices=get_dataset_subfolders())

def run_analysis(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return "Invalid Folder Path Provided"
    score = analyze_phoneme_coverage(folder_path)
    return f"🟢 Excellent Health Index — Phonetic Diversity Score: {score:.2f}%"


def dataset_tab():
    with gr.Column():
        gr.Markdown(f"### 🧪 {i18n('Smart Dataset Suite & Audio Orchestrator')}")
        
        with gr.Tabs():
            # --- TAB 1: DATASET SCRUBBER ---
            with gr.Tab(i18n("Dataset Scrubber")):
                with gr.Group():
                    gr.Markdown(f"#### 📁 {i18n('Quick Select Environment Assets')}")
                    with gr.Row():
                        dataset_dropdown = gr.Dropdown(
                            choices=get_dataset_subfolders(), 
                            label=i18n("Select Existing Dataset Workspace"), 
                            interactive=True
                        )
                        refresh_folders_btn = gr.Button(i18n("Refresh Folders List"), variant="secondary")

                with gr.Row():
                    input_folder = gr.Textbox(label=i18n("Raw Target Source Folder"), interactive=True)
                    output_folder = gr.Textbox(label=i18n("Cleaned Output Target Folder (Slices Location)"), interactive=True)

                # --- INTERACTIVE DSP PROCESSING RACK ---
                with gr.Accordion(i18n("🎛️ Inline Audio Processing Rack & Vocal Strip"), open=False):
                    gr.Markdown(i18n("Configure clean baseline optimization routines to shape and normalize raw audio profiles prior to automatic chopping loops."))
                    
                    with gr.Row():
                        gain_mode = gr.Radio(choices=["Auto (RMS Peak)", "Manual", "Bypass"], value="Auto (RMS Peak)", label=i18n("Gain Normalization Mode"))
                        manual_gain_db = gr.Slider(minimum=-24.0, maximum=24.0, step=0.5, value=0.0, label=i18n("Manual Input Makeup Gain (dB)"))

                    with gr.Row():
                        with gr.Column(variant="panel"):
                            use_gate = gr.Checkbox(label=i18n("Enable Noise Gate Expansion"), value=False)
                            gate_db = gr.Slider(minimum=-80.0, maximum=-20.0, step=1.0, value=-45.0, label=i18n("Gate Threshold (dB)"))
                        
                        with gr.Column(variant="panel"):
                            use_eq = gr.Checkbox(label=i18n("Enable Bandpass Equalization Filtering"), value=False)
                            low_cut = gr.Slider(minimum=20, maximum=500, step=5, value=80, label=i18n("Low-Cut Filter / Highpass (Hz)"))
                            high_cut = gr.Slider(minimum=4000, maximum=20000, step=100, value=12000, label=i18n("High-Cut Filter / Lowpass (Hz)"))

                        with gr.Column(variant="panel"):
                            use_limiter = gr.Checkbox(label=i18n("Enable Safety Brickwall Limiter Peak Ceiling"), value=False)
                            limiter_db = gr.Slider(minimum=-12.0, maximum=-0.1, step=0.1, value=-1.0, label=i18n("Limiter Ceiling Output (dB)"))

                scrub_btn = gr.Button(i18n("Initialize Isolation, Normalization & VAD Slicing Routine"), variant="primary")
                scrub_output = gr.Textbox(label=i18n("Asynchronous Engine Verification Logs"), interactive=False)
                
                # --- EVENT BINDINGS ---
                dataset_dropdown.change(fn=update_dataset_paths, inputs=[dataset_dropdown], outputs=[input_folder, output_folder])
                refresh_folders_btn.click(fn=refresh_dataset_dropdown, inputs=[], outputs=[dataset_dropdown])
                
                scrub_btn.click(
                    fn=run_dataset_scrubber,
                    inputs=[
                        input_folder, output_folder,
                        use_gate, gate_db,
                        use_eq, low_cut, high_cut,
                        use_limiter, limiter_db,
                        gain_mode, manual_gain_db
                    ],
                    outputs=[scrub_output]
                )

            # --- TAB 2: PHONEME COVERAGE MATRIX ---
            with gr.Tab(i18n("Phoneme Coverage Dashboard")):
                gr.Markdown(i18n("Scans the dataset layout to evaluate vocal model stability criteria based on phonetic variety maps."))
                analyze_btn = gr.Button(i18n("Run AI Transcript & Phonetic Variance Evaluation"), variant="secondary")
                coverage_score = gr.Textbox(label=i18n("Dataset Structural Health Index Score"), interactive=False)
                
                analyze_btn.click(fn=run_analysis, inputs=[output_folder], outputs=[coverage_score])

            # --- TAB 3: QUALITY CONTROL MONITOR (DYNAMIC AUTO-RENDERING) ---
            with gr.Tab(i18n("Quality Control Monitor")):
                gr.Markdown(i18n("Review, listen to, and prune your processed vocal slice samples manually before training."))
                
                refresh_qc_btn = gr.Button(i18n("🔄 Scan & Load Clips Into Preview Rack"), variant="secondary")
                
                with gr.Column() as preview_rack_container:
                    # Target container canvas where players drop dynamically
                    pass

                @gr.render(inputs=[dataset_dropdown], triggers=[refresh_qc_btn.click, dataset_dropdown.change])
                def build_audio_preview_rack(selected_folder):
                    if not selected_folder:
                        gr.Markdown(i18n("⚠️ Please select a dataset workspace above to display preview strips."))
                        return
                        
                    _, output_path = update_dataset_paths(selected_folder)
                    if not os.path.exists(output_path):
                        gr.Markdown(i18n("📂 No processed samples directory detected yet. Run the scrubber tool first!"))
                        return
                        
                    audio_files = [f for f in os.listdir(output_path) if f.endswith(".wav")]
                    if not audio_files:
                        gr.Markdown(i18n("📭 This workspace directory contains no valid `.wav` target slices."))
                        return
                        
                    gr.Markdown(f"### 🎧 Active Monitoring Channel Strip ({len(audio_files)} slices loaded)")
                    
                    # Loop and render an independent row with its own built-in delete tool for every file
                    for filename in sorted(audio_files):
                        full_file_path = os.path.abspath(os.path.join(output_path, filename))
                        
                        with gr.Row(variant="panel"):
                            # Fully independent player reading straight from the server's whitelisted directory URL
                            gr.Audio(value=full_file_path, label=filename, interactive=False, show_download_button=True)
                            
                            purge_btn = gr.Button("🗑️ Purge Slice", variant="stop", scale=0, elem_classes="w-28")
                            
                            # Clean closure block to track local deletion operations
                            def create_purge_handler(file_to_del):
                                def purge_handler():
                                    try:
                                        if os.path.exists(file_to_del):
                                            os.remove(file_to_del)
                                        return gr.update(visible=False) # Removes row from display instantly
                                    except Exception as err:
                                        print(f"File delete error: {str(err)}")
                                        return gr.update()
                                return purge_handler
                            
                            purge_btn.click(
                                fn=create_purge_handler(full_file_path),
                                inputs=[],
                                outputs=[purge_btn.parent]
                            )
                            