from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Load dataset and apply preprocessing for model training."""
    df = pd.read_csv(csv_path)

    # Remove columns that are not part of unsupervised feature training.
    df = df.drop(columns=["transaction_id", "is_fraud"], errors="ignore")

    # Keep only numeric columns for Isolation Forest.
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns available after preprocessing.")

    if "amount" in numeric_df.columns:
        numeric_df["amount"] = np.log1p(numeric_df["amount"].clip(lower=0))

    # Fill missing numeric values with median to keep feature distributions stable.
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    return numeric_df


def train_and_save_model() -> None:
    backend_dir = Path(__file__).resolve().parent
    dataset_path = (backend_dir / ".." / "dataset" / "credit_card_fraud_10k.csv").resolve()
    model_path = backend_dir / "model.pkl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    numeric_df = load_and_preprocess_data(dataset_path)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    model = IsolationForest(random_state=42, contamination="auto")
    model.fit(scaled_data)

    # Save fitted components plus feature columns for dynamic inference alignment.
    artifact = {"scaler": scaler, "model": model, "feature_columns": list(numeric_df.columns)}
    joblib.dump(artifact, model_path)

    print(f"Model trained on {len(numeric_df)} rows.")
    print(f"Saved model artifact to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
