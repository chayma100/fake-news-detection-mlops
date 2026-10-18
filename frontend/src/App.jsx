import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "https://fake-news-detection-mlops.onrender.com";

const EXAMPLES = [
  {
    label: "Political news",
    text: "The Senate voted on Tuesday to approve a bipartisan spending bill that would fund the federal government through the end of the fiscal year, averting a shutdown. The measure passed after weeks of negotiations between Democratic and Republican leaders, and now heads to the House for a final vote.",
  },
  {
    label: "Sensational claim",
    text: "BREAKING: Shocking leaked video proves the media has been lying to you for years, and they don't want you to see what happens next. Share this before it gets deleted, because the establishment is desperate to hide the truth from the American people!",
  },
  {
    label: "Absurd claim",
    text: "Scientists confirm the earth is flat according to secret documents released by anonymous insiders, who say that governments around the world have hidden the truth from ordinary citizens for decades.",
  },
];

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function Gauge({ value, label }) {
  const offset = CIRCUMFERENCE * (1 - value);
  return (
    <div className={`gauge ${label}`}>
      <svg viewBox="0 0 120 120" width="140" height="140">
        <circle className="gauge-bg" cx="60" cy="60" r={RADIUS} />
        <circle
          className="gauge-fg"
          cx="60"
          cy="60"
          r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-value">{(value * 100).toFixed(1)}%</span>
        <span className="gauge-caption">confidence</span>
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const enough = wordCount >= 20;

  async function analyze() {
    setError("");
    setResult(null);
    if (!enough) {
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
      setError(`Could not reach the API (${e.message}). Please try again.`);
    } finally {
      setLoading(false);
    }
  }

  function useExample(example) {
    setText(example.text);
    setResult(null);
    setError("");
  }

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <main className="shell">
        <header className="hero">
          <span className="badge">NLP · MLOps project</span>
          <h1>
            Fake News <span className="gradient">Detector</span>
          </h1>
          <p className="subtitle">
            Paste a news article and a machine learning model will estimate whether
            it looks reliable or fake.
          </p>
        </header>

        <section className="card">
          <div className="examples">
            <span className="examples-label">Try an example:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                className="chip"
                onClick={() => useExample(ex)}
                type="button"
              >
                {ex.label}
              </button>
            ))}
          </div>

          <textarea
            rows={9}
            placeholder="Paste an article here (English, at least 20 words)..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="row">
            <span className={`count ${enough ? "ok" : ""}`}>
              {wordCount} words {enough ? "✓" : "(min. 20)"}
            </span>
            <div className="actions">
              {text && (
                <button
                  className="ghost"
                  type="button"
                  onClick={() => {
                    setText("");
                    setResult(null);
                    setError("");
                  }}
                >
                  Clear
                </button>
              )}
              <button className="primary" onClick={analyze} disabled={loading}>
                {loading ? "Analyzing..." : "Analyze article"}
              </button>
            </div>
          </div>

          {loading && (
            <div className="loading">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
              <p>
                Contacting the API. The first call can take up to a minute if the
                server was asleep.
              </p>
            </div>
          )}

          {error && <p className="error">{error}</p>}

          {result && (
            <div className={`result ${result.label}`}>
              <Gauge value={result.probability} label={result.label} />
              <div className="result-body">
                <span className="verdict">
                  {result.label === "fake" ? "Likely FAKE" : "Likely REAL"}
                </span>
                <p>
                  {result.label === "fake"
                    ? "The writing style and vocabulary resemble articles labeled as unreliable in the training data."
                    : "The writing style and vocabulary resemble articles labeled as reliable in the training data."}
                </p>
              </div>
            </div>
          )}
        </section>

        <footer className="footer">
          <p>
            Model: TF-IDF + logistic regression trained on the WELFake dataset
            (mostly US political news). It learns writing style, not facts, so it
            may generalize poorly to other topics, styles or languages. This is a
            demo, not a fact-checker.
          </p>
        </footer>
      </main>
    </div>
  );
}