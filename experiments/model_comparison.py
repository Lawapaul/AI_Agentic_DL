import time
import pandas as pd
import numpy as np
from data.loader import IDSDataLoader
from models.trainer import IDSModelTrainer
from sklearn.metrics import f1_score

MODEL_LIST = ["cnn", "resnet", "lstm", "gru", "transformer", "hybrid"]

def run_experiment():
    print("\n=== MODEL COMPARISON EXPERIMENT ===\n")

    # Load data once
    loader = IDSDataLoader()
    data = loader.load_and_preprocess()

    results = []

    for model_type in MODEL_LIST:
        print(f"\n==============================")
        print(f"Training {model_type.upper()}")
        print(f"==============================")

        trainer = IDSModelTrainer(model_type=model_type)

        start_train = time.time()

        trainer.train(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            epochs=3,              # Keep small for comparison
            batch_size=256         # Reduce GPU memory pressure
        )

        train_time = time.time() - start_train

        # Evaluation
        metrics = trainer.evaluate(data['X_test'], data['y_test'])

        # F1 Macro
        y_pred = trainer.predict(data['X_test'], return_probabilities=False)
        f1 = f1_score(data['y_test'], y_pred, average="macro")

        # Inference timing
        start_inf = time.time()
        _ = trainer.predict(data['X_test'][:1000])
        inf_time = (time.time() - start_inf) / 1000

        # Parameter count
        params = trainer.model.count_params()

        results.append({
            "model": model_type,
            "accuracy": metrics["accuracy"],
            "f1_macro": f1,
            "train_time_sec": train_time,
            "inference_time_per_sample_sec": inf_time,
            "parameters": params
        })

    df = pd.DataFrame(results)
    df.to_csv("model_comparison_results.csv", index=False)

    print("\n=== FINAL RESULTS ===")
    print(df)
    print("\nSaved to model_comparison_results.csv")

if __name__ == "__main__":
    run_experiment()