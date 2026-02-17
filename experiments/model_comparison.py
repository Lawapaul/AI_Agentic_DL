import time
import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from models.trainer import IDSModelTrainer


MODEL_LIST = ["cnn", "lstm", "gru", "hybrid", "resnet", "transformer"]


def load_cicids2017_dataset(path="cicids2017/*.csv", max_per_class=30000):
    print("\n=== Loading CICIDS2017 Dataset ===\n")

    files = glob.glob(path)
    print("Found files:", len(files))

    df_list = []

    for file in files:
        print("Loading:", file)
        temp = pd.read_csv(file)
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    print("\nOriginal Shape:", df.shape)
    print("Total Classes:", df["Label"].nunique())
    print(df["Label"].value_counts())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove inf and NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print("\nAfter cleaning:", df.shape)

    # Balanced Sampling (≈500K total)
    df = shuffle(df, random_state=42)

    df = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max_per_class))
    )

    print("\nAfter balanced sampling:", df.shape)
    print(df["Label"].value_counts())

    return df


def preprocess_data(df):
    print("\n=== Preprocessing Data ===\n")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Encoded classes:", dict(zip(le.classes_, range(len(le.classes_)))))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # Reshape for 1D models
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_classes": len(np.unique(y_encoded))
    }


def run_experiment():
    print("\n=== MODEL COMPARISON EXPERIMENT (CICIDS2017) ===\n")

    # Load & preprocess
    df = load_cicids2017_dataset()
    data = preprocess_data(df)

    results = []

    for model_type in MODEL_LIST:
        print("\n==============================")
        print(f"Training {model_type.upper()}")
        print("==============================")

        trainer = IDSModelTrainer(model_type=model_type)

        start_train = time.time()

        trainer.train(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            epochs=2,
            batch_size=64  # reduced to prevent GPU OOM
        )

        train_time = time.time() - start_train

        # Evaluation
        metrics = trainer.evaluate(data["X_test"], data["y_test"])

        # F1 Macro
        y_pred = trainer.predict(data["X_test"], return_probabilities=False)
        f1 = f1_score(data["y_test"], y_pred, average="macro")

        # Inference timing
        start_inf = time.time()
        _ = trainer.predict(data["X_test"][:1000])
        inf_time = (time.time() - start_inf) / 1000

        params = trainer.model.count_params()

        results.append({
            "model": model_type,
            "accuracy": metrics["accuracy"],
            "f1_macro": f1,
            "train_time_sec": train_time,
            "inference_time_per_sample_sec": inf_time,
            "parameters": params
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("model_comparison_results.csv", index=False)

    print("\n=== FINAL RESULTS ===")
    print(df_results)
    print("\nSaved to model_comparison_results.csv")


if __name__ == "__main__":
    run_experiment()
