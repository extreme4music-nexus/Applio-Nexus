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
                blend_transients              = args_list.pop()  # New slider element
                blend_prosody                 = args_list.pop()  # New slider element
                blend_timbre                  = args_list.pop()  # Swapped slider element
                performance_vibrato_intensity = args_list.pop()
                performance_vibrato_style     = args_list.pop()
                performance_breathiness       = args_list.pop()
                performance_grit              = args_list.pop()
                f0_method_b                   = args_list.pop()
                index_file_b                  = args_list.pop()
                model_file_b                  = args_list.pop()

                # Secure explicit type casting for numeric elements
                blend_bias                    = float(blend_bias)
                blend_timbre                  = float(blend_timbre)
                blend_prosody                 = float(blend_prosody)
                blend_transients              = float(blend_transients)
                performance_vibrato_intensity = float(performance_vibrato_intensity)
                performance_breathiness       = float(performance_breathiness)
                performance_grit              = float(performance_grit)
                blend_velocity_switching      = bool(blend_velocity_switching)

                # --- CRITICAL TRACK TRUNCATION ---
                # run_infer_script strictly expects up to 59 foundational position arguments. Force-truncating here:
                base_args = list(args_list[:59])

                # --- UNIFIED FEATURE-LEVEL MORPH PAYLOAD ---
                # Instead of allocating file write locks across two parallel render tracks,
                # we bundle the morph modifiers and forward them to a single execution chain pass.
                nexus_kwargs = {
                    "model_path_b": model_file_b if model_file_b else None,
                    "index_path_b": index_file_b if index_file_b else "",
                    "f0_method_b": f0_method_b,
                    "performance_grit": performance_grit,
                    "performance_breathiness": performance_breathiness,
                    "performance_vibrato_style": performance_vibrato_style,
                    "performance_vibrato_intensity": performance_vibrato_intensity,
                    "blend_timbre": blend_timbre,
                    "blend_prosody": blend_prosody,
                    "blend_transients": blend_transients,
                    "blend_velocity_switching": blend_velocity_switching,
                    "blend_bias": blend_bias,
                    "embedder_model_b": embedder_model_b,
                    "embedder_model_custom_b": embedder_model_custom_b,
                }

                # --- SINGLE-PASS INFERENCE ENGINE FLOW ---
                print("Executing single-pass Feature-Level Neural Morph Matrix loop...")
                res = run_infer_script(*base_args, **nexus_kwargs)
        
                # Unpack result cleanly based on wrapper architecture expectations
                output_message = res[0] if isinstance(res, tuple) else "Inference process complete."
                output_path = res[1] if isinstance(res, tuple) else base_args[6] # Index 6 maps to output_path

                return output_message, output_path

            except Exception:
                traceback.print_exc()
                return "An error occurred during real-time feature matrix morph processing.", None