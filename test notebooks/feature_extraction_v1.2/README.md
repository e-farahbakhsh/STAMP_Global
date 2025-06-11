This version can generate PNG files of reconstructed maps, with selected feature values plotted along trench lines, using parallel processing for improved performance. To run the notebook with a specific model, simply set the model name in the `parameters` file.

If the `plate_model_name` parameter is set to `None`, the model will be loaded from the `plate_model` folder. Please ensure that all required files are present in this folder before executing the notebook.

Additionally, verify that both `time_span` and `temporal_resolution` are correctly configured in the `parameters` file.
