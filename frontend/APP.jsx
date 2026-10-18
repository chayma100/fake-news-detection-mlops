import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "https://fake-news-detection-mlops.onrender.com";

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  async function analyze() {
    setError("");
    setResult(null);
    if (wordCount < 20) {
      setError("Please paste a longer text (at least 20 words).");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ articles: [text] }),
      });
      if (!res.ok) throw new Error(`API returned status ${res.status}`);
      const data = await res.json();
      setResult(data.predictions[0]);
    } catch (e) {
      setError(`API error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Fake News Detector</h1>
      <p className="note">
        TF-IDF + logistic regression trained on WELFake (mostly US political news).
        It may generalize poorly to other topics, styles or languages.
      </p>

      <textarea
        rows={10}
        placeholder="Paste an article (English, at least 20 words)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="row">
        <span className="count">{wordCount} words</span>
        <button onClick={analyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {loading && (
        <p className="note">
          Contacting the API. The first call can take up to a minute if it was asleep.
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className={`result ${result.label}`}>
          <strong>
            Predicted: {result.label === "fake" ? "FAKE" : "REAL"}
          </strong>
          <span> (confidence {(result.probability * 100).toFixed(1)}%)</span>
          <div className="bar">
            <div style={{ width: `${result.probability * 100}%` }} />
          </div>
        </div>
      )}
    </main>
  );
}