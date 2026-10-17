import re

# marqueurs de source repérés par l'EDA
SOURCE_MARKERS = ["reuters", "getty images", "featured image", "read more", "new york times"]

def clean_text(s: str, scrub_sources: bool) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)   # URLs
    s = re.sub(r"<.*?>", " ", s)                   # balises HTML
    if scrub_sources:
        for m in SOURCE_MARKERS:
            s = s.replace(m, " ")
    s = re.sub(r"[^a-z\s]", " ", s)                # garde les lettres
    return re.sub(r"\s+", " ", s).strip()