import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out

def top_features(model, vec, k=25):
    names = np.array(vec.get_feature_names_out())
    if hasattr(model, "coef_"):
        score = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        score = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return None
    order = np.argsort(score)
    return ("Mots les plus associés à real (label 0) :\n" + ", ".join(names[order[:k]])
            + "\n\nMots les plus associés à fake (label 1) :\n" + ", ".join(names[order[-k:]]))

def main():
    params = yaml.safe_load(open("params.yaml"))
    X_test = sparse.load_npz("data/processed/X_test.npz")
    y_test = pd.read_csv("data/processed/y_test.csv")["label"]
    model = joblib.load("models/model.pkl")
    vec = joblib.load("models/tfidf_vectorizer.pkl")

    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["real", "fake"])
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    variant = "scrubbed" if params["preprocessing"]["scrub_sources"] else "raw"
    run_name = f"{params['model']['type']}_{variant}_{params['features']['max_features']}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(params))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("models/tfidf_vectorizer.pkl")
        tf = top_features(model, vec)
        if tf:
            mlflow.log_text(tf, "top_features.txt")

    print(run_name, {k: round(v, 4) for k, v in metrics.items()})

if __name__ == "__main__":
    main()