import pandas as pd
import yaml
from src.text_cleaning import clean_text

TITLE_COL, TEXT_COL, LABEL_COL = "title", "text", "label"

def preprocess(df: pd.DataFrame, scrub_sources: bool, min_words: int) -> pd.DataFrame:
    df = df.copy()
    df["content"] = (df[TITLE_COL].fillna("") + " " + df[TEXT_COL]).map(
        lambda s: clean_text(s, scrub_sources)
    )
    df = df[df["content"].str.split().str.len() >= min_words]
    return df[["content", LABEL_COL]]

def main():
    p = yaml.safe_load(open("params.yaml"))["preprocessing"]
    for split in ["train", "test"]:
        df = pd.read_csv(f"data/interim/{split}.csv")
        out = preprocess(df, p["scrub_sources"], p["min_words"])
        out.to_csv(f"data/interim/{split}_clean.csv", index=False)
        print(split, out.shape, out[LABEL_COL].value_counts().to_dict())

if __name__ == "__main__":
    main()