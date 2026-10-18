# Fake News Detection: an end-to-end MLOps project

Classifies news articles as **fake** or **real**. The focus is the MLOps workflow around the model: a reproducible pipeline, experiment tracking, a tested API, containerization, CI and deployment.

- **Frontend (React):** FRONTEND_URL
- **API (FastAPI) docs:** https://fake-news-detection-mlops.onrender.com/docs

> Both services run on free tiers and sleep after inactivity. The first request can take up to a minute.

![Real result](docs/demo-real.png)
![Fake result](docs/demo-fake.png)

## Architecture

```
WELFake CSV (DVC) -> ingestion -> preprocessing -> TF-IDF -> model -> evaluation -> MLflow
                                                              |
                                                       app/artifacts/
                                                              |
React frontend  ->  FastAPI (Docker, Render)  ->  prediction
```

| Component | Tool |
|---|---|
| Reproducible pipeline | DVC (5 stages, `dvc.yaml`, `params.yaml`) |
| Experiment tracking | MLflow (parameters, metrics, artifacts) |
| Model | TF-IDF + logistic regression (scikit-learn 1.7.2) |
| API | FastAPI, validated with Pydantic |
| Tests | pytest (4 tests) |
| Container | Docker |
| CI | GitHub Actions: tests, then Docker build |
| Deployment | Render (API as a Docker web service, frontend as a static site) |
| Frontend | React + Vite |

## Dataset

WELFake (72,134 articles, `title`, `text`, `label`). Label `1` = fake, `0` = real. After removing missing texts and duplicates: 62,718 articles, split 80/20 and stratified (train 49,711 / test 12,443 after cleaning).

## Results (MLflow)

| Run | Accuracy | Precision | Recall | F1 (fake) |
|---|---|---|---|---|
| logreg, sources removed (**final model**) | 0.9312 | 0.9237 | 0.9211 | **0.9224** |
| logreg, sources kept | 0.9436 | 0.9377 | 0.9350 | 0.9364 |
| naive Bayes, sources removed | 0.8411 | 0.8116 | 0.8362 | 0.8237 |

![MLflow comparison](docs/mlflow-compare.png)

## What I learned from the data

- **Duplicates removed before the split.** Almost all duplicated articles were fake (about 9,560 fake vs 420 real removed). Without this, copies could leak from train to test and inflate scores.
- **Source bias.** Markers like "Reuters" appear in 61.8% of real articles but only 1.8% of fake ones. I compared a run keeping these markers with a run removing them: keeping them adds only about 1.4 F1 points, so the model does not rely mainly on this shortcut. I deployed the version without them.
- **No strong overfitting.** On the run keeping the markers, F1 is 0.9465 on train vs 0.9364 on test (about 1 point).
- **Naive Bayes is clearly weaker** (F1 0.82), which justifies logistic regression.

## Limitations

- The corpus is mostly **US political news**. Generalization to other topics, writing styles or languages is unknown.
- The model learns **writing style and vocabulary, not facts**. It cannot verify a claim, so this is a demo, not a fact-checker.
- Only 5 exact source markers are removed. The style specific to each source remains.
- One run per configuration, so a 1.4-point gap is likely real but not statistically confirmed.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

dvc repro                            # runs the 5-stage pipeline and logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000

pytest                               # API tests
uvicorn app.main:app --port 8000     # http://localhost:8000/docs

docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api

cd frontend && npm install && npm run dev
```

The raw CSV is tracked by DVC and is not in Git: download `WELFake_Dataset.csv` from Kaggle into `data/raw/`.

## Updating the deployed model

The deployed API uses the two files in `app/artifacts/`. After retraining, copy `models/model.pkl` and `models/tfidf_vectorizer.pkl` there, then commit and push. Render rebuilds the image.

## Possible next steps

Bigrams experiment, pipeline tests, an F1 threshold check in CI, MLflow Model Registry, a DVC remote, hosted MLflow.