        def enforce_terms(terms_accepted, *args):
            # Initialize list immediately within function scope to eliminate UnboundLocalError
            args_list = list(args)

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
                blend_crossover_freq          = float(blend_crossover_freq)
                performance_vibrato_intensity = float(performance_vibrato_intensity)
                performance_breathiness       = float(performance_breathiness)
                performance_grit              = float(performance_grit)
                blend_velocity_switching      = bool(blend_velocity_switching)

                # --- CRITICAL TRACK TRUNCATION ---
                # run_infer_script strictly expects up to 59 arguments. Force-truncating here:
                base_args = list(args_list[:59])
                original_output_path = base_args[6] # Index 6 maps to output_path

                # Build independent target paths to prevent concurrent worker file-write locking
                base_dir = os.path.dirname(original_output_path)
                file_name = os.path.basename(original_output_path)
                path_worker_a = os.path.join(base_dir, f"worker_a_{file_name}")
                path_worker_b = os.path.join(base_dir, f"worker_b_{file_name}")

                # Task Allocation Worker A (Primary Model Configuration)
                args_worker_a = list(base_args)
                args_worker_a[6] = path_worker_a

                # Task Allocation Worker B (Secondary Model/Nexus Configuration)
                args_worker_b = list(base_args)
                args_worker_b[6] = path_worker_b
        
                # Neutralize the positional f0_method slot (Index 4) to prevent multiple-values errors
                args_worker_b[4] = None 

                # Construct Dialogue Performance Engine payload configuration
                nexus_kwargs = {
                    "net_g_path": model_file_b if model_file_b else None,
                    "file_index_path": index_file_b if index_file_b else "",
                    "f0_method": f0_method_b,
                    "performance_grit": performance_grit,
                    "performance_breathiness": performance_breathiness,
                    "performance_vibrato_style": performance_vibrato_style,
                    "performance_vibrato_intensity": performance_vibrato_intensity,
                    "blend_crossover_freq": blend_crossover_freq,
                    "blend_velocity_switching": blend_velocity_switching,
                    "blend_bias": blend_bias,
                    "embedder_model": embedder_model_b,
                    "embedder_model_custom": embedder_model_custom_b
                }

                # Multi-Worker Parallel Dispatch Engine Execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_a = executor.submit(run_infer_script, *args_worker_a)

                    if model_file_b:
                        future_b = executor.submit(run_infer_script, *args_worker_b, **nexus_kwargs)
                    else:
                        future_b = None

                    # Retrieve independent rendering tracks
                    res_a = future_a.result()
                    actual_path_a = res_a[1] if isinstance(res_a, tuple) else path_worker_a
    
                    if future_b:
                        res_b = future_b.result()
                        actual_path_b = res_b[1] if isinstance(res_b, tuple) else path_worker_b

                        # Audio Blending Layer (Using librosa array extraction)
                        audio_a, sr = librosa.load(actual_path_a, sr=None)
                        audio_b, _ = librosa.load(actual_path_b, sr=sr)

                        # Clamp array lengths to avoid buffer mismatch errors
                        min_len = min(len(audio_a), len(audio_b))
                        audio_a, audio_b = audio_a[:min_len], audio_b[:min_len]

                        # Composite mix linear sum mapping
                        blended_audio = (audio_a * (1.0 - blend_bias)) + (audio_b * blend_bias)

                        # Write back master file array
                        sf.write(original_output_path, blended_audio, sr)

                        # Cleanup detached worker disk cache footprint
                        for path_temp in [actual_path_a, actual_path_b]:
                            if os.path.exists(path_temp):
                                os.remove(path_temp)
        
                        return "Parallel execution and blending complete.", original_output_path

                    # Fallback if no Model B is designated by user
                    if os.path.exists(path_worker_a):
                        if os.path.exists(original_output_path):
                            os.remove(original_output_path)
                        shutil.move(path_worker_a, original_output_path)
                    return "Primary model inference complete.", original_output_path

            except Exception:
                traceback.print_exc()
                return "An error occurred during parallel synthesis.", None