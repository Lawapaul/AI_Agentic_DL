"""
Data loading and preprocessing module for CICIDS2017 dataset.

This module handles:
1. Multi-CSV dataset loading
2. Strict preprocessing (academic compliant)
3. Balanced 500k sampling
4. Train-test split
5. Reshaping for 1D CNN
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class IDSDataLoader:
    """Handles loading and preprocessing of CICIDS2017 dataset."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = None

    # ============================================================
    # LOAD CICIDS2017 (ALL CSV FILES)
    # ============================================================
    def load_dataset(self):

        print("Loading CICIDS2017 dataset...")

        possible_paths = [
            "/kaggle/input/cicids2017",
            "/content/cicids2017",
            os.path.expanduser("~/.cache/kagglehub/datasets/cicids2017")
        ]

        dataset_path = None

        for path in possible_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csv_files:
                    dataset_path = path
                    print(f"Found dataset at: {path}")
                    break

        if dataset_path is None:
            raise FileNotFoundError(
                "CICIDS2017 dataset not found. Download it using kagglehub."
            )

        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

        print(f"Found {len(csv_files)} CSV files")
        print("Merging CSV files...")

        df_list = []
        for file in csv_files:
            file_path = os.path.join(dataset_path, file)
            print(f"Loading {file}")
            df_list.append(pd.read_csv(file_path, low_memory=False))

        df = pd.concat(df_list, ignore_index=True)

        print(f"Merged dataset shape: {df.shape}")
        return df

    # ============================================================
    # PREPROCESSING
    # ============================================================
    def preprocess(self, df):

        print("\n=== Starting Preprocessing ===")

        # Step 1: Separate label
        if 'Label' in df.columns:
            label_col = 'Label'
        else:
            label_col = df.columns[-1]

        y = df[label_col].copy()
        X = df.drop(columns=[label_col])

        print(f"Samples: {len(y)}")
        print(f"Initial feature count: {X.shape[1]}")
        print(f"Unique attack types: {y.nunique()}")
        print("Attack distribution:\n", y.value_counts())

        # Step 2: Convert features to numeric
        print("\nConverting features to numeric...")
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Step 3: Drop fully-NaN columns
        X = X.dropna(axis=1, how='all')
        print("Feature count after drop:", X.shape[1])

        # Step 4: Drop NaN rows
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        print("After dropping NaN rows:", X.shape)

        # Step 5: Replace infinity
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        print("After removing infinity rows:", X.shape)

        # Step 6: Clip extreme values
        for col in X.columns:
            lower = X[col].quantile(0.001)
            upper = X[col].quantile(0.999)
            X[col] = X[col].clip(lower, upper)

        self.feature_names = X.columns.tolist()

        # Step 7: Encode labels
        print("\nEncoding labels...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.label_mapping = {
            idx: label for idx, label in enumerate(self.label_encoder.classes_)
        }

        print("Label mapping:", self.label_mapping)

        # Step 8: Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        print(
            f"Scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}"
        )

        # ============================================================
        # STEP 9: BALANCED SAMPLING (500K TOTAL)
        # ============================================================
        print("\nApplying balanced sampling (500k total)...")

        df_balanced = pd.DataFrame(X_scaled)
        df_balanced["label"] = y_encoded

        num_classes = len(np.unique(y_encoded))
        samples_per_class = 500000 // num_classes

        print(f"Classes: {num_classes}")
        print(f"Target per class: {samples_per_class}")

        balanced_df = (
            df_balanced.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), samples_per_class),
                random_state=42
            ))
        )

        print("Balanced dataset shape:", balanced_df.shape)

        y_balanced = balanced_df["label"].values
        X_balanced = balanced_df.drop(columns=["label"]).values

        return X_balanced, y_balanced

    # ============================================================
    # SPLIT
    # ============================================================
    def train_test_split_data(self, X, y):

        print("\nSplitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)

        return X_train, X_test, y_train, y_test

    # ============================================================
    # RESHAPE FOR CNN
    # ============================================================
    def reshape_for_cnn(self, X_train, X_test):

        print("\nReshaping for CNN...")

        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], 1
        )
        X_test = X_test.reshape(
            X_test.shape[0], X_test.shape[1], 1
        )

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        return X_train, X_test

    # ============================================================
    # FULL PIPELINE
    # ============================================================
    def load_and_preprocess(self):

        df = self.load_dataset()
        X, y = self.preprocess(df)
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)
        X_train, X_test = self.reshape_for_cnn(X_train, X_test)

        print("\n=== Preprocessing Complete ===")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'label_mapping': self.label_mapping,
            'scaler': self.scaler,
            'num_features': len(self.feature_names),
            'num_classes': len(self.label_mapping)
        }


def load_and_preprocess():
    loader = IDSDataLoader()
    return loader.load_and_preprocess()


if __name__ == "__main__":
    print("Testing CICIDS2017 Data Loader...")
    data = load_and_preprocess()

    print("\n=== Data Summary ===")
    print("Training samples:", data['X_train'].shape[0])
    print("Test samples:", data['X_test'].shape[0])
    print("Features:", data['num_features'])
    print("Classes:", data['num_classes'])
    print("Label mapping:", data['label_mapping'])