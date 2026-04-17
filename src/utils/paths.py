from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
RAW_IMAGES_ANCHOR = "data/raw/images/"


def portable_image_path(value: str | Path) -> Path:
    """
    Resolve image paths saved on one machine so they still work on another.

    The CSVs in this project may contain:
    - relative repo paths like data/raw/images/ISIC_....jpg
    - absolute Windows paths from the author's machine
    """
    raw = str(value)
    path = Path(raw)

    if path.exists():
        return path

    normalized = raw.replace("\\", "/")

    if normalized.startswith(RAW_IMAGES_ANCHOR):
        candidate = REPO_ROOT / Path(normalized)
        if candidate.exists():
            return candidate

    if RAW_IMAGES_ANCHOR in normalized:
        suffix = normalized.split(RAW_IMAGES_ANCHOR, 1)[1]
        candidate = RAW_IMAGES_DIR / suffix
        if candidate.exists():
            return candidate

    fallback = RAW_IMAGES_DIR / path.name
    if fallback.exists():
        return fallback

    return path


def repo_relative_image_path(image_id: str) -> str:
    return (Path("data") / "raw" / "images" / f"{image_id}.jpg").as_posix()
