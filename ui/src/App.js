import React, { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ label: "Error", probability: 0 });
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Fake Job Posting Detector</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste job description and requirements here..."
          rows={8}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </form>
      {result && (
        <div className={`result ${result.label}`}>
          <h2>
            Prediction:{" "}
            <span className={result.label}>
              {result.label === "fraudulent" ? "🚩 Fraudulent" : "✅ Legitimate"}
            </span>
          </h2>
          <p>
            Probability: <strong>{(result.probability * 100).toFixed(2)}%</strong>
          </p>
        </div>
      )}
      <footer>
        <p>
          Powered by DeBERTa-v3-base &middot; <a href="https://github.com/your-repo">GitHub</a>
        </p>
      </footer>
    </div>
  );
}

export default App;