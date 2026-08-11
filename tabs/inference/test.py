            embedder_model_b = gr.Radio(
                label=i18n("Embedder Model B"),
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
                with gr.Accordion(i18n("Custom Embedder B"), open=True):
                    with gr.Row():
                        embedder_model_custom_b = gr.Dropdown(
                            label=i18n("Select Custom Embedder"),
                            choices=refresh_embedders_folders(),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        refresh_embedders_button_b = gr.Button(i18n("Refresh embedders"))
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