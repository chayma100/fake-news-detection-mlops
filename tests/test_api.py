from fastapi.testclient import TestClient
from app.main import app
from src.text_cleaning import clean_text

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_predict():
    with TestClient(app) as client:
        r = client.post("/predict", json={"articles": ["Some news text here"]})
        assert r.status_code == 200
        pred = r.json()["predictions"][0]
        assert pred["label"] in ("fake", "real")
        assert 0.0 <= pred["probability"] <= 1.0

def test_predict_rejects_empty_list():
    with TestClient(app) as client:
        r = client.post("/predict", json={"articles": []})
        assert r.status_code == 422

def test_clean_text_removes_source_markers():
    assert "reuters" not in clean_text("WASHINGTON (Reuters) - Hello", True)