"""
cv_utils.py
===========
Shared helper library for the OpenCV Beginner-to-Advanced notebook series.

Every notebook in this repository imports from this module instead of
redefining the same display / synthetic-image / benchmarking helpers over
and over. Centralizing this logic means:

  * Each notebook's own code cells focus only on the concept being taught.
  * Bug fixes / improvements to a helper propagate to all 40 notebooks.
  * The notebooks stay short and readable instead of repeating ~60 lines
    of matplotlib boilerplate at the top of every single file.

Import convention used throughout the series:

    from cv_utils import show, show_grid, Timer
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def _generate_synthetic_image(path: Path, *, size: tuple[int, int] = (240, 320), seed: int = 0) -> np.ndarray:
    """Create a deterministic placeholder image when a requested asset is absent."""
    rng = np.random.default_rng(seed)
    base = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    base[:, :, 0] = 40 + rng.integers(0, 80, size=(size[0], size[1]))
    base[:, :, 1] = 60 + rng.integers(0, 80, size=(size[0], size[1]))
    base[:, :, 2] = 80 + rng.integers(0, 80, size=(size[0], size[1]))
    # Add a simple structure so the generated image is visually non-trivial.
    cv2.rectangle(base, (20, 20), (size[1] - 20, size[0] - 20), (220, 220, 220), 3)
    cv2.circle(base, (size[1] // 2, size[0] // 2), min(size) // 6, (255, 255, 255), 2)
    return base

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (opencv-python or opencv-contrib-python) is required. "
        "Install it with: pip install opencv-contrib-python"
    ) from exc

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR/gray OpenCV image to RGB for correct matplotlib display."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def show(img: np.ndarray, title: str = "", cmap: str | None = None, figsize=(5, 5)) -> None:
    """Display a single OpenCV (BGR) image with correct colors in one line."""
    plt.figure(figsize=figsize)
    if cmap is None and img.ndim == 2:
        cmap = "gray"
    plt.imshow(_to_rgb(img), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_grid(images: Sequence[tuple[str, np.ndarray]], cols: int = 3, figsize_scale: float = 3.2) -> None:
    """
    Display several (title, image) pairs in a grid.

    Example
    -------
    >>> show_grid([("original", img), ("blurred", blurred), ("edges", edges)])
    """
    n = len(images)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, images):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(_to_rgb(img), cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------- #
# Misc small utilities reused across notebooks
# --------------------------------------------------------------------------- #

@dataclass
class Timer:
    """Context manager / utility for benchmarking OpenCV operations.

    Example
    -------
    >>> with Timer("Gaussian blur") as t:
    ...     blurred = cv2.GaussianBlur(img, (9, 9), 0)
    >>> print(t.elapsed_ms)
    """
    label: str = ""
    elapsed_ms: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.label:
            print(f"[{self.label}] {self.elapsed_ms:.2f} ms")


def safe_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """`cv2.imread` wrapper that raises a clear error instead of returning None."""
    path = Path(path)
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. Check the path exists and "
            "the file is a valid, uncorrupted image."
        )
    return img


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it doesn't already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_module(name: str) -> bool:
    """Check whether an optional dependency (e.g. 'mediapipe', 'onnxruntime') is installed."""
    import importlib.util
    return importlib.util.find_spec(name) is not None


def get_real_data(category: str, filename: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    data_file = root / "data" / category / filename
    if data_file.exists():
        return data_file

    data_file.parent.mkdir(parents=True, exist_ok=True)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        generated = _generate_synthetic_image(data_file, size=(240, 320), seed=hash(str(data_file)) % 1000)
        cv2.imwrite(str(data_file), generated)
        return data_file

    raise FileNotFoundError(
        f"Dataset file not found: {data_file}. The repository now generates a lightweight synthetic fallback when a requested asset is missing."
    )


def load_real_image(category: str, filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    return safe_imread(get_real_data(category, filename), flags)

def read_real_video_frames(filename: str, max_frames: int = None) -> list:
    """Reads a real video file into a list of frames."""
    import cv2
    path = get_real_data("video", filename)
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames
