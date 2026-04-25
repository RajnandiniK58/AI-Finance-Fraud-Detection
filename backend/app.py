from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)
CORS(app)


def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "scaler" not in artifact:
        raise ValueError("Invalid model artifact. Expected keys: 'model' and 'scaler'.")

    return artifact


def parse_input_to_dataframe() -> pd.DataFrame:
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            raise ValueError("Uploaded file is empty.")
        return pd.read_csv(uploaded_file)

    data = request.get_json()
    if data is None:
        raise ValueError("No valid input provided. Send a CSV file or JSON body.")

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON format. Expected an object with a 'data' key.")

    if "data" not in data:
        raise ValueError("JSON body must include a 'data' field.")

    if not isinstance(data["data"], list):
        raise ValueError("'data' must be a list of records.")

    df = pd.DataFrame(data["data"])
    if df.empty:
        raise ValueError("No rows found in JSON 'data'.")

    return df


def preprocess_for_inference(raw_df: pd.DataFrame, scaler: Any) -> Any:
    if raw_df.empty:
        raise ValueError("Input data is empty.")

    # Mirror training preprocessing exactly:
    # drop non-feature columns, then keep only numeric columns, then fill missing values.
    cleaned_df = raw_df.drop(columns=["transaction_id", "is_fraud"], errors="ignore")
    numeric_df = cleaned_df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns found in input data.")

    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    return numeric_df


def align_and_scale_features(numeric_df: pd.DataFrame, scaler: Any, feature_columns: list[str]) -> Any:
    expected_columns = list(feature_columns)
    received_columns = list(numeric_df.columns)

    missing = [col for col in expected_columns if col not in received_columns]
    extra = [col for col in received_columns if col not in expected_columns]
    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Expected columns: {expected_columns}. Received columns: {received_columns}."
        )

    aligned_df = numeric_df[expected_columns]

    expected_features = getattr(scaler, "n_features_in_", len(expected_columns))
    if aligned_df.shape[1] != expected_features:
        raise ValueError(
            f"Feature count mismatch. Model expects {expected_features}, got {aligned_df.shape[1]}."
        )

    return scaler.transform(aligned_df)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        artifact = load_artifact()
        model = artifact["model"]
        scaler = artifact["scaler"]
        feature_columns = artifact.get("feature_columns")
        if not feature_columns and hasattr(scaler, "feature_names_in_"):
            feature_columns = list(scaler.feature_names_in_)
        if not feature_columns:
            raise ValueError("Model artifact is missing training feature columns.")

        raw_df = parse_input_to_dataframe()
        numeric_df = preprocess_for_inference(raw_df, scaler)
        scaled_data = align_and_scale_features(numeric_df, scaler, feature_columns)

        raw_predictions = model.predict(scaled_data)
        # IsolationForest outputs -1 for anomalies and 1 for normal points.
        predictions = [1 if pred == -1 else 0 for pred in raw_predictions]

        fraud_count = sum(predictions)
        total = len(predictions)

        return jsonify({"predictions": predictions, "fraud_count": fraud_count, "total": total}), 200
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 500
    except Exception as err:  # noqa: BLE001
        return jsonify({"error": f"Internal server error: {err}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
