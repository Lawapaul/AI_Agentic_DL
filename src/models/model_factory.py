def get_model(model_name, input_shape, num_classes):
    if model_name == "cnn":
        from src.models.cnn_model import build_ids_cnn_model
        return build_ids_cnn_model(input_shape, num_classes)

    elif model_name == "hybrid":
        from src.models.hybrid_cnn_lstm import build_model
        return build_model(input_shape, num_classes)

    elif model_name == "lstm":
        from src.models.lstm_model import build_lstm_model
        return build_lstm_model(input_shape, num_classes)

    elif model_name == "gru":
        from src.models.gru_model import build_gru_model
        return build_gru_model(input_shape, num_classes)

    elif model_name == "resnet":
        from src.models.resnet_1d import build_resnet_model
        return build_resnet_model(input_shape, num_classes)

    elif model_name == "transformer":
        from src.models.transformer_1d import build_transformer_model
        return build_transformer_model(input_shape, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_name}")
