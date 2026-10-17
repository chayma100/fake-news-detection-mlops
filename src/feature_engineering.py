from pathlib import Path
import joblib
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    p = yaml.safe_load(open("params.yaml"))["features"]
    train = pd.read_csv("data/interim/train_clean.csv")
    test = pd.read_csv("data/interim/test_clean.csv")

    vec = TfidfVectorizer(max_features=p["max_features"],
                          ngram_range=tuple(p["ngram_range"]), stop_words="english")
    X_train = vec.fit_transform(train["content"])   # fit sur le train uniquement
    X_test = vec.transform(test["content"])

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    sparse.save_npz("data/processed/X_train.npz", X_train)
    sparse.save_npz("data/processed/X_test.npz", X_test)
    train[["label"]].to_csv("data/processed/y_train.csv", index=False)
    test[["label"]].to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(vec, "models/tfidf_vectorizer.pkl")
    print("X_train", X_train.shape, "X_test", X_test.shape)

if __name__ == "__main__":
    main()