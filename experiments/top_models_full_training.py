import time
import pandas as pd
from data.loader import IDSDataLoader
from models.trainer import IDSModelTrainer
from sklearn.metrics import f1_score

TOP_MODELS = ["hybrid", "resnet"]

def run_full_training(df):

    print("\n=== FULL DATASET TRAINING (NO SAMPLING) ===\n")

    loader = IDSDataLoader()
    X, y = loader.preprocess(df)   # use already merged df
    X_train, X_test, y_train, y_test = loader.split(X, y)
    X_train, X_test = loader.reshape(X_train, X_test)

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    print("Train size:", data['X_train'].shape)
    print("Test size:", data['X_test'].shape)

    results = []

    for model_type in TOP_MODELS:

        print("\n==============================")
        print(f"Training {model_type.upper()} on FULL DATASET")
        print("==============================")

        trainer = IDSModelTrainer(model_type=model_type)

        start_time = time.time()

        trainer.train(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            epochs=20,      # Start with 20 (safer)
            batch_size=128  # Adjust if OOM
        )

        train_time = time.time() - start_time

        metrics = trainer.evaluate(data['X_test'], data['y_test'])
        y_pred = trainer.predict(data['X_test'], return_probabilities=False)
        f1 = f1_score(data['y_test'], y_pred, average="macro")

        results.append({
            "model": model_type,
            "accuracy": metrics["accuracy"],
            "f1_macro": f1,
            "train_time_sec": train_time,
            "parameters": trainer.model.count_params()
        })

    df = pd.DataFrame(results)
    df.to_csv("experiments/results/top_models_full_20epoch.csv", index=False)

    print("\nResults saved successfully.")

if __name__ == "__main__":
    raise RuntimeError(
        "Pass a pre-merged dataframe to run_full_training(df) to avoid re-reading CSV files."
    )
