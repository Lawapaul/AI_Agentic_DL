from models.cnn_model import build_ids_cnn_model
from models.hybrid_cnn_lstm import build_hybrid_model

def get_model(model_type, input_shape, num_classes):

    if model_type == "cnn":
        return build_ids_cnn_model(input_shape, num_classes)

    elif model_type == "hybrid":
        return build_hybrid_model(input_shape, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")