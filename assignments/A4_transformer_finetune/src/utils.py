import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save a dictionary to JSON with indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def save_markdown(path: Path, content: str) -> None:
    """Save a markdown string to disk."""
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
