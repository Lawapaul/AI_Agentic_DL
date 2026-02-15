"""
Data loading and preprocessing module for IDS dataset.

This module handles:
1. Dataset loading from local cache
2. Strict preprocessing following academic requirements
3. Train-test split and reshaping for 1D CNN
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class IDSDataLoader:
    """Handles loading and preprocessing of IDS intrusion dataset."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.label_mapping = None
        
    def load_dataset(self):
        """
        Load IDS dataset from multiple possible locations.
        """

        print("Loading IDS dataset...")

        possible_paths = [
            os.path.expanduser("~/.cache/kagglehub/datasets/solarmainframe/ids-intrusion-csv/versions/1"),
            "/kaggle/input/ids-intrusion-csv",
            "/content/ids-intrusion-csv",
            "/content"
        ]

        dataset_path = None

        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset path: {path}")
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError(
                "Dataset not found in any known location.\n"
                "Please download using kagglehub before running pipeline."
            )

        # Find CSV file
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found inside {dataset_path}"
            )

        csv_path = os.path.join(dataset_path, csv_files[0])
        print(f"Loading CSV file: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

        return df
    def preprocess(self, df):
        """
        Preprocess the dataset following strict requirements:
        1. Separate Label column FIRST
        2. Convert all features to numeric
        3. Drop fully-NaN columns
        4. Drop rows with NaN
        5. Encode labels
        6. Handle infinity and extreme values
        7. Scale features
        
        Args:
            df: Raw dataframe
            
        Returns:
            tuple: (X, y) preprocessed features and labels
        """
        print("\n=== Starting Preprocessing ===")
        
        # Step 1: Separate Label column FIRST
        if 'Label' not in df.columns and 'label' not in df.columns:
            # Try to find label column (case-insensitive)
            label_cols = [col for col in df.columns if 'label' in col.lower()]
            if label_cols:
                label_col = label_cols[0]
            else:
                # Assume last column is label
                label_col = df.columns[-1]
                print(f"Warning: No 'Label' column found, using '{label_col}' as label")
        else:
            label_col = 'Label' if 'Label' in df.columns else 'label'
        
        y = df[label_col].copy()
        X = df.drop(columns=[label_col])
        
        print(f"Separated labels: {len(y)} samples")
        print(f"Feature columns: {X.shape[1]}")
        print(f"Unique attack types: {y.nunique()}")
        print(f"Attack distribution:\n{y.value_counts()}")
        
        # Step 2: Convert all feature columns to numeric
        print("\nConverting features to numeric...")
        original_cols = X.shape[1]
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        print(f"Converted {original_cols} columns to numeric")
        
        # Step 3: Drop fully-NaN columns
        print("\nDropping fully-NaN columns...")
        X = X.dropna(axis=1, how='all')
        print(f"Columns after dropping fully-NaN: {X.shape[1]}")
        
        # Step 4: Drop rows with NaN
        print("\nDropping rows with NaN values...")
        original_rows = X.shape[0]
        
        # Align X and y before dropping
        valid_indices = X.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        print(f"Rows before: {original_rows}, after: {X.shape[0]}")
        print(f"Dropped {original_rows - X.shape[0]} rows with NaN")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"\nFinal feature count: {len(self.feature_names)}")
        
        # Step 5: Encode labels using LabelEncoder
        print("\nEncoding labels...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store label mapping for interpretability
        self.label_mapping = {
            idx: label for idx, label in enumerate(self.label_encoder.classes_)
        }
        print(f"Label mapping: {self.label_mapping}")
        
        # Step 6: Handle infinity and extreme values
        print("\nHandling infinity and extreme values...")
        # Replace infinity with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop any new NaN rows created by inf replacement
        if X.isna().any().any():
            # Create boolean mask for valid rows
            valid_mask = ~X.isna().any(axis=1)
            rows_before = len(X)
            X = X[valid_mask]
            y_encoded = y_encoded[valid_mask.values]  # Use boolean mask for numpy array
            print(f"Dropped {rows_before - len(X)} rows with infinity values")
        
        # Clip extreme values to prevent scaling issues
        # Use 99.9th percentile as upper bound and 0.1th percentile as lower bound
        for col in X.columns:
            lower = X[col].quantile(0.001)
            upper = X[col].quantile(0.999)
            X[col] = X[col].clip(lower, upper)
        
        print(f"Cleaned data shape: {X.shape}")
        
        # Step 7: Scale features using StandardScaler
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features scaled: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
        
        return X_scaled, y_encoded
    
    def train_test_split_data(self, X, y):
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data: {100*(1-self.test_size):.0f}% train, {100*self.test_size:.0f}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def reshape_for_cnn(self, X_train, X_test):
        """
        Reshape data for 1D CNN input: (samples, features, 1)
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: Reshaped (X_train, X_test)
        """
        print("\nReshaping for 1D CNN...")
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"Train shape: {X_train_reshaped.shape}")
        print(f"Test shape: {X_test_reshaped.shape}")
        
        return X_train_reshaped, X_test_reshaped
    
    def load_and_preprocess(self):
        """
        Complete pipeline: load, preprocess, split, and reshape.
        
        Returns:
            dict: Contains all preprocessed data and metadata
        """
        # Load dataset
        df = self.load_dataset()
        
        # Preprocess
        X, y = self.preprocess(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)
        
        # Reshape for CNN
        X_train_cnn, X_test_cnn = self.reshape_for_cnn(X_train, X_test)
        
        print("\n=== Preprocessing Complete ===")
        
        return {
            'X_train': X_train_cnn,
            'X_test': X_test_cnn,
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
    """Convenience function for loading and preprocessing data."""
    loader = IDSDataLoader()
    return loader.load_and_preprocess()


if __name__ == "__main__":
    # Test the data loading pipeline
    print("Testing IDS Data Loader...")
    data = load_and_preprocess()
    
    print("\n=== Data Summary ===")
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Number of features: {data['num_features']}")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Feature names (first 10): {data['feature_names'][:10]}")
    print(f"Label mapping: {data['label_mapping']}")
