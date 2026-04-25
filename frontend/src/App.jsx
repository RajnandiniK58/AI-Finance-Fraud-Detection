import React, { useState } from "react";

const API_URL = "http://127.0.0.1:5000/predict";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [fraudCount, setFraudCount] = useState(0);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
    </div>
  );
}

export default App;
