import re
from pathlib import Path
import pandas as pd

TITLE_COL, TEXT_COL, LABEL_COL = "title", "text", "label"

# marqueurs repérés par l'EDA
SOURCE_MARKERS = ["reuters", "getty images", "featured image",
                  "read more", "new york times"]

def clean_text(s: str, scrub_sources: bool = True) -> str:
    s = s.lower()
    s = re.sub(r"https?://\\S+|www\\.\\S+", " ", s)   # URLs
    s = re.sub(r"<.*?>", " ", s)                     # balises HTML
    if scrub_sources:
        for m in SOURCE_MARKERS:
            s = s.replace(m, " ")
    s = re.sub(r"[^a-z\\s]", " ", s)                  # garde les lettres (dataset en anglais)
    return re.sub(r"\\s+", " ", s).strip()

def preprocess(df: pd.DataFrame, scrub_sources: bool = True) -> pd.DataFrame:
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df.drop_duplicates(subset=[TEXT_COL])            # AVANT le split train/test
    df["content"] = (df[TITLE_COL].fillna("") + " " + df[TEXT_COL]).map(
        lambda s: clean_text(s, scrub_sources)
    )
    df = df[df["content"].str.split().str.len() >= 20]    # retire les textes quasi vides
    return df[["content", LABEL_COL]]

if __name__ == "__main__":
    csv_path = next(Path("data/raw").glob("*.csv"))
    raw = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"], errors="ignore")
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    for scrub, name in [(True, "clean_scrubbed.csv"), (False, "clean_raw.csv")]:
        out = preprocess(raw.copy(), scrub_sources=scrub)
        out.to_csv(f"data/interim/{name}", index=False)
        print(name, out.shape, out[LABEL_COL].value_counts().to_dict())