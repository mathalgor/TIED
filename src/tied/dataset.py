"""TIED dataset loader for the layout:

    train/source/{real,aug}/<stem>.<ext>
    train/outline/{real,aug}/<stem>.<ext>
    test/source/<stem>.<ext>
    test/outline/<stem>.<ext>

Source-channel mode (``source = "rgb"|"gray"``) and outline encoding
(``outline = "mono"|"gray"``) come from ``tied.toml`` via
``tied.config.load_config``.

Outline encoding (edge bright on dark background, opposite of MCED):
  * "mono" — read as gray, threshold at 128 → {0., 1.} float target.
  * "gray" — read as gray, scaled to [0, 1] float target (TEED-style).

Returns dict samples ``{"source": Tensor[C,H,W], "outline": Tensor[1,H,W],
"stem": str}``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
JPEG_EXTS = {".jpg", ".jpeg"}
JPEG_BLUR_SIGMA = 0.8   # 3x3 Gaussian to wash out JPEG block artefacts


def _list_stems(d: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def _read_source(path: Path, source_mode: str) -> np.ndarray:
    if source_mode == "rgb":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read source image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cannot read source image: {path}")
    # JPEG augs (written by tied-augment) carry block artefacts that
    # the model would learn as fake edges. A small Gaussian wipes them
    # out without smudging real edges meaningfully (sigma=0.8 ~ 1 px).
    # PNGs (typically real/) are kept sharp.
    if path.suffix.lower() in JPEG_EXTS:
        img = cv2.GaussianBlur(img, (3, 3), JPEG_BLUR_SIGMA)
    if img.ndim == 2:
        return img[..., None]  # (H, W, 1) uint8
    return img  # (H, W, 3) uint8


def _read_outline(path: Path, outline_mode: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read outline image: {path}")
    if outline_mode == "mono":
        return (img >= 128).astype(np.float32)  # (H, W) {0., 1.}
    return img.astype(np.float32) / 255.0       # (H, W) [0, 1]


def _random_crop(src: np.ndarray, out: np.ndarray, size: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    h, w = src.shape[:2]
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        src = cv2.copyMakeBorder(src, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=0)
        out = cv2.copyMakeBorder(out, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=0)
        h, w = src.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return src[y:y + size, x:x + size], out[y:y + size, x:x + size]


class TiedDataset(Dataset):
    """Pair source/outline images by stem.

    ``splits`` selects which subfolders under train/ to pull from. For
    test set, pass ``flat=True`` and the root will be treated as
    test/ directly (no real/aug split).
    """

    def __init__(
        self,
        root: Path,
        source_mode: str,
        outline_mode: str,
        splits: Sequence[str] = ("real", "aug"),
        crop_size: int | None = None,
        flat: bool = False,
    ):
        self.source_mode = source_mode
        self.outline_mode = outline_mode
        self.crop_size = crop_size

        pairs: list[tuple[Path, Path, str]] = []
        if flat:
            src_dir = root / "source"
            out_dir = root / "outline"
            src_stems = _list_stems(src_dir)
            out_stems = _list_stems(out_dir)
            for stem, sp in src_stems.items():
                op = out_stems.get(stem)
                if op is not None:
                    pairs.append((sp, op, stem))
        else:
            for split in splits:
                src_dir = root / "source" / split
                out_dir = root / "outline" / split
                src_stems = _list_stems(src_dir)
                out_stems = _list_stems(out_dir)
                for stem, sp in src_stems.items():
                    op = out_stems.get(stem)
                    if op is not None:
                        pairs.append((sp, op, stem))
        if not pairs:
            raise RuntimeError(
                f"no paired source/outline images under {root} "
                f"(flat={flat}, splits={list(splits)})")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sp, op, stem = self.pairs[idx]
        src = _read_source(sp, self.source_mode)          # (H, W, C) uint8
        out = _read_outline(op, self.outline_mode)        # (H, W) float32

        if self.crop_size is not None:
            src, out = _random_crop(src, out, self.crop_size)

        src_t = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0   # (C, H, W)
        out_t = torch.from_numpy(out).unsqueeze(0).float()               # (1, H, W)
        return {"source": src_t, "outline": out_t, "stem": stem}
