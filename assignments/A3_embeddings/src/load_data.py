from typing import Optional

from nltk.corpus import gutenberg


def load_gutenberg_text(file_id: str = "austen-emma.txt") -> Optional[str]:
    """Load text from the NLTK Gutenberg corpus."""
    try:
        return gutenberg.raw(file_id)
    except LookupError:
        print(
            "NLTK data not found. Please run: "
            "python -m nltk.downloader punkt punkt_tab gutenberg"
        )
        return None
    except OSError:
        print(f"File ID not found in Gutenberg corpus: {file_id}")
        return None
