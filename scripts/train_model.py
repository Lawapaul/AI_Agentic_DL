"""Starter script for model training using the existing training module."""

from src.models.trainer import IDSModelTrainer


def main() -> None:
    trainer = IDSModelTrainer(model_type="hybrid", model_save_path="saved_models/ids_model.keras")
    print(
        "Training is still handled by the existing trainer module.\n"
        "Load data with src.utils.data_loader.IDSDataLoader and call trainer.train(...) from a notebook or script."
    )
    print(f"Configured trainer: {trainer.__class__.__name__} -> {trainer.model_save_path}")


if __name__ == "__main__":
    main()
