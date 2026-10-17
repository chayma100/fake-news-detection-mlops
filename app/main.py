import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse, Prediction
from src.text_cleaning import clean_text   # même nettoyage que l'entraînement

SCRUB_SOURCES = True    # doit correspondre au modèle déployé (params.yaml)
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["vec"] = joblib.load("app/artifacts/tfidf_vectorizer.pkl")
    ml["model"] = joblib.load("app/artifacts/model.pkl")
    yield
    ml.clear()

app = FastAPI(title="Fake News Detection API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])   # à restreindre en production

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = [clean_text(a, SCRUB_SOURCES) for a in req.articles]
    X = ml["vec"].transform(cleaned)
    proba = ml["model"].predict_proba(X)[:, 1]     # probabilité de la classe 1 = fake
    preds = [
        Prediction(label="fake" if p >= 0.5 else "real",
                   probability=float(p if p >= 0.5 else 1 - p))
        for p in proba
    ]
    return PredictResponse(predictions=preds)