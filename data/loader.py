"""
CICIDS2017 Data Loading & Preprocessing Module

Features:
1. Multi-CSV automatic merging
2. Strict academic preprocessing
3. Optional balanced sampling
4. Optional full dataset usage
5. GPU-optimized float32 casting
6. Clean train-test split
7. 1D reshape for DL models
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class IDSDataLoader:
    """
    Handles loading and preprocessing of CICIDS2017 dataset.
    """

    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        balanced_total_samples=500000,   # Set None for full dataset
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.balanced_total_samples = balanced_total_samples

        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = None

    # ============================================================
    # LOAD ALL CSV FILES
    # ============================================================
    # ============================================================
# LOAD ALL CSV FILES
# ============================================================
    def load_dataset(self):

        print("Loading CICIDS2017 dataset...")

        # Primary expected path (Colab project structure)
        dataset_path = os.path.join(os.getcwd(), "data", "raw", "cicids2017")

        # Fallback paths (in case someone runs differently)
        fallback_paths = [
            os.path.join(os.getcwd(), "cicids2017"),
            "/content/cicids2017",
            "/content/data/raw/cicids2017",
        ]

        if not os.path.exists(dataset_path):
            for path in fallback_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"CICIDS dataset not found. Checked:\n"
                f" - {os.path.join(os.getcwd(), 'data/raw/cicids2017')}\n"
                f" - fallback locations"
            )

        print(f"Dataset path: {dataset_path}")

        csv_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in dataset directory.")

        print(f"Found {len(csv_files)} CSV files")
        print("Merging files...")

        df_list = []
        for file in csv_files:
            print("Loading:", file)
            df_list.append(pd.read_csv(file, low_memory=False))

        df = pd.concat(df_list, ignore_index=True)

        print("Original shape:", df.shape)
        return df

    # ============================================================
    # PREPROCESSING
    # ============================================================
    def preprocess(self, df):

        print("\n=== Starting Preprocessing ===")

        label_col = "Label" if "Label" in df.columns else df.columns[-1]

        y = df[label_col]
        X = df.drop(columns=[label_col])

        print("Initial samples:", len(y))
        print("Initial features:", X.shape[1])
        print("Unique classes:", y.nunique())

        # Fast numeric conversion
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop fully NaN columns
        X = X.dropna(axis=1, how="all")

        # Remove rows with NaN or Inf in ONE step (fast)
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]

        print("After cleaning:", X.shape)

        # ============================================================
        # BALANCED SAMPLING
        # ============================================================
        if self.balanced_total_samples is not None:

            print("\nApplying balanced sampling...")

            temp_df = X.copy()
            temp_df["Label"] = y

            class_counts = temp_df["Label"].value_counts()
            num_classes = len(class_counts)

            samples_per_class = self.balanced_total_samples // num_classes

            balanced_list = []

            for label in class_counts.index:
                class_subset = temp_df[temp_df["Label"] == label]
                n_samples = min(samples_per_class, len(class_subset))

                balanced_list.append(
                    class_subset.sample(
                        n=n_samples,
                        random_state=self.random_state
                    )
                )

            balanced_df = pd.concat(balanced_list)

            # Shuffle
            balanced_df = balanced_df.sample(
                frac=1,
                random_state=self.random_state
            )

            y = balanced_df["Label"]
            X = balanced_df.drop(columns=["Label"])

            print("After balanced sampling:", X.shape)


        self.feature_names = X.columns.tolist()

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.label_mapping = {
            idx: label for idx, label in enumerate(self.label_encoder.classes_)
        }

        print("Label mapping done.")

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_scaled = X_scaled.astype(np.float32)
        y_encoded = y_encoded.astype(np.int32)

        print("Scaling complete.")

        return X_scaled, y_encoded

    # ============================================================
    # SPLIT
    # ============================================================
    def split(self, X, y):

        print("\nSplitting dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        return X_train, X_test, y_train, y_test

    # ============================================================
    # RESHAPE FOR 1D MODELS
    # ============================================================
    def reshape(self, X_train, X_test):

        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], 1
        )
        X_test = X_test.reshape(
            X_test.shape[0], X_test.shape[1], 1
        )

        return X_train, X_test

    # ============================================================
    # FULL PIPELINE
    # ============================================================
    def load_and_preprocess(self):

        df = self.load_dataset()
        X, y = self.preprocess(df)
        X_train, X_test, y_train, y_test = self.split(X, y)
        X_train, X_test = self.reshape(X_train, X_test)

        print("\n=== Preprocessing Complete ===")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "label_mapping": self.label_mapping,
            "scaler": self.scaler,
            "num_features": len(self.feature_names),
            "num_classes": len(self.label_mapping),
        }


# Convenience function
def load_and_preprocess(balanced_total_samples=500000):
    loader = IDSDataLoader(
        balanced_total_samples=balanced_total_samples
    )
    return loader.load_and_preprocess()


if __name__ == "__main__":
    print("Testing Loader...")
    data = load_and_preprocess()

    print("\nTraining samples:", data["X_train"].shape[0])
    print("Test samples:", data["X_test"].shape[0])
    print("Classes:", data["num_classes"])
