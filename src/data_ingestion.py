from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def main():
    p = yaml.safe_load(open("params.yaml"))["data"]
    df = pd.read_csv(p["raw_path"]).drop(columns=["Unnamed: 0"], errors="ignore")
    n0 = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])      # AVANT le split : évite la fuite de données
    print(f"{n0} -> {len(df)} lignes après suppression des manquants et doublons")

    train, test = train_test_split(
        df, test_size=p["test_size"], random_state=p["random_state"], stratify=df["label"]
    )
    out = Path("data/interim")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print("train", train.shape, "test", test.shape)

if __name__ == "__main__":
    main()