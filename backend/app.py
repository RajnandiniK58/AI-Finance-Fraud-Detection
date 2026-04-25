from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = Flask(__name__)
CORS(app)


def get_inference_components() -> tuple[Any, Any, list[str]]:
    artifact = load_artifact()
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact.get("feature_columns")
    if not feature_columns and hasattr(scaler, "feature_names_in_"):
        feature_columns = list(scaler.feature_names_in_)
    if not feature_columns:
        raise ValueError("Model artifact is missing training feature columns.")
    return model, scaler, list(feature_columns)


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
    alias_map = {
        "transaction_hour": "transaction",
        "device_trust_score": "device_trust",
        "velocity_last_24h": "velocity_last_hour",
    }
    for source_col, target_col in alias_map.items():
        if source_col in cleaned_df.columns and target_col not in cleaned_df.columns:
            cleaned_df[target_col] = cleaned_df[source_col]
    numeric_df = cleaned_df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns found in input data.")

    if "amount" in numeric_df.columns:
        numeric_df["amount"] = np.log1p(numeric_df["amount"].clip(lower=0))

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


def compute_fraud_confidence(anomaly_score: float, raw_row: pd.Series) -> float:
    # Convert IsolationForest decision score to fraud-likelihood confidence in [0, 1].
    confidence = float(1.0 / (1.0 + np.exp(anomaly_score)))

    amount = float(raw_row.get("amount", 0.0))
    foreign_txn = float(raw_row.get("foreign_transaction", 0.0))
    velocity = float(raw_row.get("velocity_last_24h", raw_row.get("velocity_last_hour", 0.0)))

    if amount > 50000 and foreign_txn == 1:
        confidence += 0.15
    if velocity > 10:
        confidence += 0.10

    return float(np.clip(confidence, 0.0, 1.0))


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        model, scaler, feature_columns = get_inference_components()

        raw_df = parse_input_to_dataframe()
        missing_cols = set(feature_columns) - set(raw_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        raw_df = raw_df[feature_columns]
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


@app.route("/predict_single", methods=["POST"])
def predict_single() -> Any:
    try:
        data = request.get_json()
        if data is None or not isinstance(data, dict):
            raise ValueError("Invalid JSON body for single transaction prediction.")

        single_df = pd.DataFrame([data])
        model, scaler, feature_columns = get_inference_components()
        numeric_df = preprocess_for_inference(single_df, scaler)
        scaled_data = align_and_scale_features(numeric_df, scaler, feature_columns)

        raw_prediction = model.predict(scaled_data)[0]
        anomaly_score = float(model.decision_function(scaled_data)[0])
        confidence = compute_fraud_confidence(anomaly_score, single_df.iloc[0])
        prediction = 1 if raw_prediction == -1 else 0
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({"prediction": prediction, "result": result, "confidence": confidence}), 200
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 500
    except Exception as err:  # noqa: BLE001
        return jsonify({"error": f"Internal server error: {err}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
