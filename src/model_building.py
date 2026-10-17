import joblib
import pandas as pd
import yaml
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def main():
    p = yaml.safe_load(open("params.yaml"))["model"]
    X_train = sparse.load_npz("data/processed/X_train.npz")
    y_train = pd.read_csv("data/processed/y_train.csv")["label"]

    if p["type"] == "logreg":
        model = LogisticRegression(C=p["logreg_C"], max_iter=p["logreg_max_iter"])
    elif p["type"] == "naive_bayes":
        model = MultinomialNB()
    else:
        raise ValueError(f"Modèle inconnu : {p['type']}")

    model.fit(X_train, y_train)
    joblib.dump(model, "models/model.pkl")
    print("Modèle entraîné :", p["type"])

if __name__ == "__main__":
    main()