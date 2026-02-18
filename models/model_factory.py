def get_model(model_name, input_shape, num_classes):
    if model_name == "cnn":
        from models.cnn_model import build_ids_cnn_model
        return build_ids_cnn_model(input_shape, num_classes)

    elif model_name == "hybrid":
        from models.hybrid_cnn_lstm import build_model
        return build_model(input_shape, num_classes)

    elif model_name == "lstm":
        from models.lstm_model import build_lstm_model
        return build_lstm_model(input_shape, num_classes)

    elif model_name == "gru":
        from models.gru_model import build_gru_model
        return build_gru_model(input_shape, num_classes)

    elif model_name == "resnet":
        from models.resnet_1d import build_resnet_model
        return build_resnet_model(input_shape, num_classes)

    elif model_name == "transformer":
        from models.transformer_1d import build_transformer_model
        return build_transformer_model(input_shape, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_name}")
