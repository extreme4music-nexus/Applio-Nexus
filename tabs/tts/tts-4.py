import json
import os
import random
import sys
import traceback
import concurrent.futures
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import shutil
from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_tts_script
from tabs.inference.inference import (
    change_choices,
    create_folder_and_move_files,
    default_weight,
    extract_model_and_epoch,
    filter_dropdowns,
    get_files,
    get_speakers_id,
    match_index,
    refresh_embedders_folders,
    update_filter_visibility,
)
from tabs.settings.sections.filter import get_filter_trigger, load_config_filter

i18n = I18nAuto()


with open(
    os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
) as file:
    tts_voices_data = json.load(file)

short_names = [voice.get("ShortName", "") for voice in tts_voices_data]


def process_input(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file.read()
        gr.Info(f"The file has been loaded!")
        return file_path, file_path
    except UnicodeDecodeError:
        gr.Info(f"The file has to be in UTF-8 encoding.")
        return None, None


def tts_tab():
    trigger = get_filter_trigger()
    
    # Pre-declare variables to allocate memory references before layout attachments occur
    model_file_b = gr.Dropdown(visible=False)
    index_file_b = gr.Dropdown(visible=False)

    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(get_files("model"), key=extract_model_and_epoch),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            filter_box_tts = gr.Textbox(
                label=i18n("Filter"),
                info=i18n("Path must contain:"),
                placeholder=i18n("Type to filter..."),
                interactive=True,
                scale=0.1,
                visible=load_config_filter(),
                elem_id="filter_box_tts",
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=sorted(get_files("index")),
                value=match_index(default_weight),
                interactive=True,
                allow_custom_value=True,
                close_on_click=True,
            )
            filter_box_tts.blur(
                fn=filter_dropdowns,
                inputs=[filter_box_tts],
                outputs=[model_file, index_file],
            )
            trigger.change(
                fn=update_filter_visibility,
                inputs=[trigger],
                outputs=[filter_box_tts, model_file, index_file],
                show_progress=False,
            )
        with gr.Row():
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file, model_file_b, index_file_b],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    gr.Markdown(
        i18n(
            f"Applio is a Speech-to-Speech conversion software, utilizing EdgeTTS as middleware for running the Text-to-Speech (TTS) component. Read more about it [here!](https://docs.applio.org/applio/getting-started/tts)"
        )
    )
    tts_voice = gr.Dropdown(
        label=i18n("TTS Voices"),
        info=i18n("Select the TTS voice to use for the conversion."),
        choices=short_names,
        interactive=True,
        value=random.choice(short_names),
    )

    tts_rate = gr.Slider(
        minimum=-100,
        maximum=100,
        step=1,
        label=i18n("TTS Speed"),
        info=i18n("Increase or decrease TTS speed."),
        value=0,
        interactive=True,
    )

    with gr.Tabs():
        with gr.Tab(label=i18n("Text to Speech")):
            tts_text = gr.Textbox(
                label=i18n("Text to Synthesize"),
                info=i18n("Enter the text to synthesize."),
                placeholder=i18n("Enter text to synthesize"),
                lines=3,
            )
        with gr.Tab(label=i18n("File to Speech")):
            txt_file = gr.File(
                label=i18n("Upload a .txt file"),
                type="filepath",
            )
            input_tts_path = gr.Textbox(
                label=i18n("Input path for text file"),
                placeholder=i18n(
                    "The path to the text file that contains content for text to speech."
                ),
                value="",
                interactive=True,
            )

    with gr.Accordion(i18n("Advanced Settings"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=i18n("Output Path for TTS Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=i18n("Output Path for RVC Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            sid = gr.Dropdown(
                label=i18n("Speaker ID"),
                info=i18n("Select the speaker ID to use for the conversion."),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n(
                    "Apply a soft autotune to your inferences, recommended for singing conversions."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Autotune Strength"),
                info=i18n(
                    "Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."
                ),
                visible=False,
                value=1,
                interactive=True,
            )
            proposed_pitch = gr.Checkbox(
                label=i18n("Proposed Pitch"),
                info=i18n(
                    "Adjust the input audio pitch to match the voice model range."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            proposed_pitch_threshold = gr.Slider(
                minimum=50.0,
                maximum=1200.0,
                label=i18n("Proposed Pitch Threshold"),
                info=i18n(
                    "Male voice models typically use 155.0 and female voice models typically use 255.0."
                ),
                visible=False,
                value=155.0,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=i18n("Clean Audio"),
                info=i18n(
                    "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Clean Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=True,
                value=0.5,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n(
                    "Set the pitch of the audio, the higher the value, the higher the pitch."
                ),
                value=0,
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n(
                    "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                ),
                value=0.75,
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Volume Envelope"),
                info=i18n(
                    "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                ),
                value=1,
                interactive=True,
            )
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n(
                    "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                ),
                value=0.5,
                interactive=True,
            )
            f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n(
                    "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                ),
                choices=[
                    "crepe",
                    "crepe-tiny",
                    "rmvpe",
                    "fcpe",
                ],
                value="rmvpe",
                interactive=True,
            )
            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=[
                    "contentvec",
                    "spin",
                    "spin-v2",
                    "chinese-hubert-base",
                    "japanese-hubert-base",
                    "korean-hubert-base",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
            )
            with gr.Column(visible=False) as embedder_custom:
                with gr.Accordion(i18n("Custom Embedder"), open=True):
                    with gr.Row():
                        embedder_model_custom = gr.Dropdown(
                            label=i18n("Select Custom Embedder"),
                            choices=refresh_embedders_folders(),
                            interactive=True,
                            allow_custom_value=True,
                            close_on_click=True,
                        )
                        refresh_embedders_button = gr.Button(i18n("Refresh embedders"))
                    folder_name_input = gr.Textbox(
                        label=i18n("Folder Name"), interactive=True
                    )
                    with gr.Row():
                        bin_file_upload = gr.File(
                            label=i18n("Upload .bin"),
                            type="filepath",
                            interactive=True,
                        )
                        config_file_upload = gr.File(
                            label=i18n("Upload .json"),
                            type="filepath",
                            interactive=True,
                        )
                    move_files_button = gr.Button(
                        i18n("Move files to custom embedder folder")
                    )
            f0_file = gr.File(
                label=i18n(
                    "The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls."
                ),
                visible=True,
            )

    with gr.Accordion(i18n("Nexus Performance & Conversational Multi-Model Matrix"), open=False):
        gr.Markdown(f"### 💬 {i18n('Dialogue Performance Expression Engine')}")
        with gr.Row():
            performance_grit = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                label=i18n("Dialogue Grit / Aggression Intensity"),
                info=i18n("Injects gravelly fold characteristics, perfect for simulating elevated emotion or anger.")
            )
            performance_breathiness = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                label=i18n("Dialogue Breathiness / Secretive Whisper"),
                info=i18n("Introduces high-frequency noise elements to give conversation an intimate, fatigued, or whispered feel.")
            )
        with gr.Row():
            performance_vibrato_style = gr.Dropdown(
                choices=["None", "Pop", "Jazz", "Opera"],
                value="None",
                label=i18n("Conversational Vibrato CAD Emulation"),
                info=i18n("Applies micro-expression fluctuations onto steady spoken notes.")
            )
            performance_vibrato_intensity = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0,
                label=i18n("Vibrato Dialogue Modulation Depth"),
                info=i18n("Adjusts the resonance limits of the underlying tracking vocal path oscillators.")
            )
        
        gr.Markdown(f"### 👥 {i18n('Conversational Multi-Model Blending Node')}")
        with gr.Row():
            model_file_b = gr.Dropdown(
                label=i18n("Voice Model B (Secondary Character Slot)"),
                info=i18n("Select a second voice model checkpoint to build dynamic conversational hybrid variations."),
                choices=sorted(get_files("model"), key=extract_model_and_epoch),
                value="",
                interactive=True,
                allow_custom_value=True,
                close_on_click=True,
            )
            index_file_b = gr.Dropdown(
                label=i18n("Index File B (Secondary Character Index)"),
                info=i18n("Select the feature database file corresponding to your secondary character."),
                choices=sorted(get_files("index")),
                value="",
                interactive=True,
                allow_custom_value=True,
                close_on_click=True,
            )
        with gr.Row():
            embedder_model_b = gr.Radio(
                label=i18n("Embedder Model B"),
                choices=["contentvec", "spin", "spin-v2", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base", "custom"],
                value="contentvec",
                interactive=True
            )
            f0_method_b = gr.Radio(
                label=i18n("Pitch extraction algorithm B"),
                info=i18n("Pitch extraction algorithm to use for Voice Model B conversion pipeline."),
                choices=["crepe", "crepe-tiny", "rmvpe", "fcpe"],
                value="rmvpe",
                interactive=True,
            )
            with gr.Column(visible=False) as embedder_custom_b_container:
                embedder_model_custom_b = gr.Dropdown(label=i18n("Select Custom Embedder B"), choices=refresh_embedders_folders(), interactive=True, close_on_click=True)
                refresh_embedders_button_b = gr.Button(i18n("Refresh embedders B"))

            embedder_model_b.change(
                fn=lambda x: {"visible": x == "custom", "__type__": "update"},
                inputs=[embedder_model_b],
                outputs=[embedder_custom_b_container]
            )  
        with gr.Row():
            blend_crossover_freq = gr.Slider(
                minimum=20.0,
                maximum=4000.0,
                step=10.0,
                value=800.0,
                label=i18n("Dialogue Crossover Cutoff (Hz)"),
                info=i18n("Allocates foundational resonance frequencies to Character A and crystal-clear speech elements to Character B.")
            )
            enable_crossover = gr.Checkbox(label=i18n("Enable Frequency Crossover"), value=True)
            crossover_mode = gr.Radio(choices=["A-Low/B-High", "B-Low/A-High"], value="A-Low/B-High", label=i18n("Crossover Phase Mapping"))
            
            blend_bias = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                label=i18n("Inter-Character Interaction Balance"),
                info=i18n("Adjusts model dominance (0.0 fully tracks Voice A, 1.0 shifts completely to Voice B).")
            )
        with gr.Row():
            blend_velocity_switching = gr.Checkbox(
                label=i18n("Conversational Volume-Triggered Line Switching"),
                value=False,
                info=i18n("Enables real-time line crossfading based on word volume (quiet turns favor Voice A, while stressed words favor Voice B).")
            )
            
    # --- COMPLETELY IMPLEMENTED INFERENCE BLENDING CORE LOGIC ---
    def enforce_terms(terms_accepted, *args):
        # Clamp the input list to exactly 40 elements to drop hidden Gradio layout footprints
        args_list = list(args[:40])
    
        if not terms_accepted:
            message = "You must agree to the Terms of Use to proceed."
            gr.Info(message)
            return message, None

        try:
            # --- EXPLICIT POP OPERATIONS FROM THE TAIL ---
            embedder_model_custom_b       = args_list.pop()
            embedder_model_b              = args_list.pop()
            blend_bias                    = args_list.pop()
            blend_velocity_switching      = args_list.pop()
            crossover_mode                = args_list.pop()
            enable_crossover              = args_list.pop()
            blend_crossover_freq          = args_list.pop()
            performance_vibrato_intensity = args_list.pop()
            performance_vibrato_style     = args_list.pop()
            performance_breathiness       = args_list.pop()
            performance_grit              = args_list.pop()
            f0_method_b                   = args_list.pop()
            index_file_b                  = args_list.pop()
            model_file_b                  = args_list.pop()

            # Secure explicit type casting for numeric elements
            blend_bias                    = float(blend_bias)
            enable_crossover              = bool(enable_crossover)
            blend_crossover_freq          = float(blend_crossover_freq)
            performance_vibrato_intensity = float(performance_vibrato_intensity)
            performance_breathiness       = float(performance_breathiness)
            performance_grit              = float(performance_grit)
            blend_velocity_switching      = bool(blend_velocity_switching)

            # --- CRITICAL TRACK TRUNCATION ---
            # run_tts_script strictly expects up to 24 positional core parameters.
            base_tts_args = list(args_list[:24])
            original_rvc_path = base_tts_args[11] # Index 11 maps to output_rvc_path in tts wrapper

            # Build independent target paths to isolate file creation loops
            base_dir = os.path.dirname(original_rvc_path)
            file_name = os.path.basename(original_rvc_path)
            path_worker_a = os.path.join(base_dir, f"tts_worker_a_{file_name}")
            path_worker_b = os.path.join(base_dir, f"tts_worker_b_{file_name}")

            # Task Allocation Worker A (Primary Model Configuration)
            args_worker_a = list(base_tts_args)
            args_worker_a[11] = path_worker_a

            # Task Allocation Worker B (Secondary Model/Nexus Configuration)
            args_worker_b = list(base_tts_args)
            args_worker_b[11] = path_worker_b
        
            # --- FIXED POSITION-BASED DIRECT OVERRIDES ---
            # Directly updating positional parameters to prevent wrapper dictionary updates from losing these keys
            args_worker_b[9] = f0_method_b              # Index 9: f0_method
            args_worker_b[11] = path_worker_b           # Index 11: output_rvc_path
            args_worker_b[12] = model_file_b            # Index 12: model_file
            args_worker_b[13] = index_file_b            # Index 13: index_file
            args_worker_b[22] = embedder_model_b        # Index 22: embedder_model
            args_worker_b[23] = embedder_model_custom_b # Index 23: embedder_model_custom

            # Construct Dialogue Performance Engine payload configuration
            nexus_kwargs = {
                "performance_grit": performance_grit,
                "performance_breathiness": performance_breathiness,
                "performance_vibrato_style": performance_vibrato_style,
                "performance_vibrato_intensity": performance_vibrato_intensity,
            }

            # --- SEQUENTIAL DISPATCH ENGINE FLOW ---
            print("Executing TTS Worker A sequential rendering loop...")
            info_text_a, actual_path_a = run_tts_script(*args_worker_a)
            if not actual_path_a or not os.path.exists(path_worker_a):
                actual_path_a = path_worker_a

            if model_file_b:
                print("Executing TTS Worker B sequential rendering loop with Nexus parameters...")
                info_text_b, actual_path_b = run_tts_script(*args_worker_b, **nexus_kwargs)
                if not actual_path_b or not os.path.exists(path_worker_b):
                    actual_path_b = path_worker_b

                # --- ADVANCED DSP BLENDING LAYER ---
                print("Applying Butterworth Crossover & Dynamic Envelope Blending...")
                audio_a, sr = librosa.load(actual_path_a, sr=None)
                audio_b, _ = librosa.load(actual_path_b, sr=sr)

                min_len = min(len(audio_a), len(audio_b))
                audio_a, audio_b = audio_a[:min_len], audio_b[:min_len]
                
                # Initialize variables to raw audio to prevent UnboundLocalError
                audio_a_filtered = audio_a
                audio_b_filtered = audio_b

                # 1. Zero-Phase Butterworth Crossover Filtering
                if enable_crossover:
                    nyquist = sr * 0.5
                    clamped_crossover = np.clip(blend_crossover_freq, 100.0, nyquist - 100.0)
                    Wn = clamped_crossover / nyquist
                    b_low, a_low = signal.butter(4, Wn, btype='low')
                    b_high, a_high = signal.butter(4, Wn, btype='high')
                    
                    # APPLY PHASE FLIP: A-Low/B-High OR B-Low/A-High
                    if crossover_mode == "A-Low/B-High":
                        audio_a_filtered = signal.filtfilt(b_low, a_low, audio_a)
                        audio_b_filtered = signal.filtfilt(b_high, a_high, audio_b)
                    else:
                        audio_a_filtered = signal.filtfilt(b_high, a_high, audio_a)
                        audio_b_filtered = signal.filtfilt(b_low, a_low, audio_b)

                # 2. Dynamic Volume-Triggered Velocity Switching
                if blend_velocity_switching:
                    rms = librosa.feature.rms(y=audio_a, frame_length=2048, hop_length=512)[0]
                    rms_norm = (rms - np.min(rms)) / (np.ptp(rms) + 1e-8)
                    times_rms = np.linspace(0, min_len, len(rms_norm))
                    dynamic_envelope = np.interp(np.arange(min_len), times_rms, rms_norm)
                    active_bias = np.clip((blend_bias * 0.4) + (dynamic_envelope * 0.6), 0.0, 1.0)
                else:
                    active_bias = blend_bias

                # 3. Final Composite Mix
                gain_a = (1.0 - active_bias) * 2.0
                gain_b = active_bias * 2.0
                blended_audio = (audio_a_filtered * gain_a) + (audio_b_filtered * gain_b)

                # Anti-clipping peak normalization
                max_amp = np.max(np.abs(blended_audio))
                if max_amp > 0.99:
                    blended_audio = (blended_audio / max_amp) * 0.99

                sf.write(original_rvc_path, blended_audio, sr)

                for path_temp in [actual_path_a, actual_path_b]:
                    if os.path.exists(path_temp):
                        os.remove(path_temp)

                return "TTS Sequential execution and crossover blending complete.", original_rvc_path

            if os.path.exists(path_worker_a):
                if os.path.exists(original_rvc_path):
                    os.remove(original_rvc_path)
                shutil.move(path_worker_a, original_rvc_path)
            return "Primary TTS model inference complete.", original_rvc_path

        except Exception:
            traceback.print_exc()
            return "An error occurred during sequential TTS synthesis processing.", None
            
    terms_checkbox = gr.Checkbox(
        label=i18n("I agree to the terms of use"),
        info=i18n(
            "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference."
        ),
        value=False,
        interactive=True,
    )
    convert_button = gr.Button(i18n("Convert"))
    
    with gr.Row():
        vc_output1 = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
        )
        vc_output2 = gr.Audio(label=i18n("Export Audio"))
        
    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}
        
    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}
        
    autotune.change(
        fn=toggle_visible,
        inputs=[autotune],
        outputs=[autotune_strength],
    )
    proposed_pitch.change(
        fn=toggle_visible,
        inputs=[proposed_pitch],
        outputs=[proposed_pitch_threshold],
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, model_file_b, sid, sid, model_file_b, index_file_b],
    ).then(
        fn=filter_dropdowns,
        inputs=[filter_box_tts],
        outputs=[model_file, index_file],
    )
    
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[input_tts_path, txt_file],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    
    model_file_b.select(
        fn=lambda model_file_value: match_index(model_file_value),
        inputs=[model_file_b],
        outputs=[index_file_b],
    )
    convert_button.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,                 #0
            input_tts_path,                 #1
            tts_text,                       #2
            tts_voice,                      #3
            tts_rate,                       #4
            pitch,                          #5
            index_rate,                     #6
            rms_mix_rate,                   #7
            protect,                        #8
            f0_method,                      #9
            output_tts_path,                #10
            output_rvc_path,                #11
            model_file,                     #12
            index_file,                     #13
            split_audio,                    #14
            autotune,                       #15
            autotune_strength,              #16
            proposed_pitch,                 #17
            proposed_pitch_threshold,       #18
            clean_audio,                    #19
            clean_strength,                 #20
            export_format,                  #21
            embedder_model,                 #22
            embedder_model_custom,          #23
            sid,                            #24
            f0_file,                        #25
            model_file_b,                   #26
            index_file_b,                   #27
            f0_method_b,                    #28
            performance_grit,               #29
            performance_breathiness,        #30
            performance_vibrato_style,      #31
            performance_vibrato_intensity,  #32
            blend_crossover_freq,           #33
            enable_crossover,               #34
            crossover_mode,                 #35
            blend_velocity_switching,       #36
            blend_bias,                     #37
            embedder_model_b,               #38
            embedder_model_custom_b,        #39
        ],
        outputs=[vc_output1, vc_output2],
    )