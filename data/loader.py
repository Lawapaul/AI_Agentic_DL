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
    def load_dataset(self):

        print("Loading CICIDS2017 dataset...")

        possible_paths = [
            "/kaggle/input/ids-intrusion-csv",
            "/kaggle/input/cicids2017",
            "/content/ids-intrusion-csv",
            "/content/cicids2017",
        ]

        dataset_path = None

        for path in possible_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csv_files:
                    dataset_path = path
                    break

        if dataset_path is None:
            raise FileNotFoundError(
                "CICIDS dataset not found. Please download it first."
            )

        print(f"Dataset path: {dataset_path}")

        csv_files = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(".csv")
        ]

        print(f"Found {len(csv_files)} CSV files")
        print("Merging files...")

        df_list = []
        for file in csv_files:
            print("Loading:", file)
            df_list.append(pd.read_csv(file, low_memory=False))

        df = pd.concat(df_list, ignore_index=True)

        print("Merged dataset shape:", df.shape)
        return df

    # ============================================================
    # PREPROCESSING
    # ============================================================
    def preprocess(self, df):

        print("\n=== Starting Preprocessing ===")

        # 1️⃣ Separate label
        label_col = "Label" if "Label" in df.columns else df.columns[-1]

        y = df[label_col]
        X = df.drop(columns=[label_col])

        print("Initial samples:", len(y))
        print("Initial features:", X.shape[1])
        print("Unique classes:", y.nunique())

        # 2️⃣ Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # 3️⃣ Remove fully NaN columns
        X = X.dropna(axis=1, how="all")

        # 4️⃣ Remove NaN rows
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # 5️⃣ Remove infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        print("After cleaning:", X.shape)

        # 6️⃣ Clip extreme values
        for col in X.columns:
            lower = X[col].quantile(0.001)
            upper = X[col].quantile(0.999)
            X[col] = X[col].clip(lower, upper)

        self.feature_names = X.columns.tolist()

        # 7️⃣ Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.label_mapping = {
            idx: label
            for idx, label in enumerate(self.label_encoder.classes_)
        }

        print("Label mapping:", self.label_mapping)

        # 8️⃣ Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # GPU optimization
        X_scaled = X_scaled.astype(np.float32)
        y_encoded = y_encoded.astype(np.int32)

        print("Scaling complete.")

        # ============================================================
        # OPTIONAL BALANCED SAMPLING
        # ============================================================
        if self.balanced_total_samples is not None:

            print("\nApplying balanced sampling...")

            df_bal = pd.DataFrame(X_scaled)
            df_bal["label"] = y_encoded

            num_classes = len(np.unique(y_encoded))
            samples_per_class = self.balanced_total_samples // num_classes

            print("Classes:", num_classes)
            print("Target per class:", samples_per_class)

            balanced_df = (
                df_bal.groupby("label", group_keys=False)
                .apply(
                    lambda x: x.sample(
                        min(len(x), samples_per_class),
                        random_state=self.random_state,
                    )
                )
            )

            X_scaled = balanced_df.drop(columns=["label"]).values
            y_encoded = balanced_df["label"].values

            print("Balanced dataset shape:", X_scaled.shape)

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