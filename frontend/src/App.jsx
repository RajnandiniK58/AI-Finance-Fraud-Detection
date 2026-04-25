import React, { useState } from "react";

const API_URL = "http://127.0.0.1:5000/predict";
const SINGLE_API_URL = "http://127.0.0.1:5000/predict_single";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [fraudCount, setFraudCount] = useState(0);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleError, setSingleError] = useState("");
  const [singleResult, setSingleResult] = useState(null);
  const [singleForm, setSingleForm] = useState({
    amount: "",
    transaction_hour: "",
    foreign_transaction: "",
    location_mismatch: "",
    device_trust_score: "",
    velocity_last_24h: "",
    cardholder_age: "",
  });

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError("Please select a CSV file.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to get predictions.");
      }

      setPredictions(Array.isArray(data.predictions) ? data.predictions : []);
      setFraudCount(Number(data.fraud_count) || 0);
      setTotal(Number(data.total) || 0);
    } catch (apiError) {
      setError(apiError.message || "API request failed.");
      setPredictions([]);
      setFraudCount(0);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  };

  const handleSingleInputChange = (event) => {
    const { name, value } = event.target;
    setSingleForm((prev) => ({ ...prev, [name]: value }));
    setSingleError("");
    setSingleResult(null);
  };

  const handleSingleSubmit = async (event) => {
    event.preventDefault();
    setSingleLoading(true);
    setSingleError("");
    setSingleResult(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(singleForm).map(([key, value]) => [key, Number(value)])
      );

      if (Object.values(payload).some((value) => Number.isNaN(value))) {
        throw new Error("Please enter valid numbers for all single transaction fields.");
      }

      const response = await fetch(SINGLE_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to check single transaction.");
      }

      setSingleResult(data);
    } catch (apiError) {
      setSingleError(apiError.message || "API request failed.");
    } finally {
      setSingleLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>AI Finance Fraud Detection</h1>

      <form className="upload-form" onSubmit={handleSubmit}>
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? "Submitting..." : "Submit"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      <div className="summary">
        <p>Total transactions: {total}</p>
        <p>Fraud count: {fraudCount}</p>
      </div>

      {predictions.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Transaction #</th>
              <th>Prediction</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((prediction, index) => (
              <tr key={index} className={prediction === 1 ? "fraud-row" : ""}>
                <td>{index + 1}</td>
                <td>{prediction}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Check Single Transaction</h2>
      <form className="single-form" onSubmit={handleSingleSubmit}>
        <input name="amount" type="number" step="any" placeholder="Amount" value={singleForm.amount} onChange={handleSingleInputChange} />
        <input
          name="transaction_hour"
          type="number"
          step="any"
          placeholder="Transaction Hour"
          value={singleForm.transaction_hour}
          onChange={handleSingleInputChange}
        />
        <input
          name="foreign_transaction"
          type="number"
          step="any"
          placeholder="Foreign Transaction (0/1)"
          value={singleForm.foreign_transaction}
          onChange={handleSingleInputChange}
        />
        <input
          name="location_mismatch"
          type="number"
          step="any"
          placeholder="Location Mismatch (0/1)"
          value={singleForm.location_mismatch}
          onChange={handleSingleInputChange}
        />
        <input
          name="device_trust_score"
          type="number"
          step="any"
          placeholder="Device Trust Score"
          value={singleForm.device_trust_score}
          onChange={handleSingleInputChange}
        />
        <input
          name="velocity_last_24h"
          type="number"
          step="any"
          placeholder="Velocity Last 24h"
          value={singleForm.velocity_last_24h}
          onChange={handleSingleInputChange}
        />
        <input
          name="cardholder_age"
          type="number"
          step="any"
          placeholder="Cardholder Age"
          value={singleForm.cardholder_age}
          onChange={handleSingleInputChange}
        />
        <button type="submit" disabled={singleLoading}>
          {singleLoading ? "Checking..." : "Check Fraud"}
        </button>
      </form>

      {singleError && <p className="error">{singleError}</p>}
      {singleResult && (
        <div>
          <p className={singleResult.prediction === 1 ? "result-fraud" : "result-safe"}>
            {singleResult.result}
          </p>
          <p>
            Confidence: {Math.round((Number(singleResult.confidence) || 0) * 100)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
