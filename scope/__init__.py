"""SCOPE - Topic modeling CLI for finding contiguous conversation blocks."""

__version__ = "0.1.0"

# Ensure required NLP data is downloaded
def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

# Download on import
try:
    _ensure_nltk_data()
except Exception:
    # Fail silently - will error later if actually needed
    pass
