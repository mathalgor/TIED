#!/usr/bin/env python3
"""Generate offline geometric augmentations for source and outline.

For each input file we write the chosen preset's variants. Tags:
  * D4 \\ {e} — bit-exact: r90, r180, r270, flipH, flipV, trans, atrans
  * 15-degree nearest-neighbour rotations (skipping the D4 angles).

Same transform applied to source and outline keeps pixels aligned, which
is important because the loss has only a small tolerance band.

Background fill for rotations / warps is 0 (black) — matches the TIED
convention that edges are bright on dark.

Usage:

    tied-augment                     # both source + outline, full preset
    tied-augment --preset portrait   # flipH + +/- 5/10/15 rotations
    tied-augment --outlines          # only outlines (skip source)

Reads dataset root and channel modes from ``tied.toml``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 7]

BG_VALUE = 0   # dark background — edges in TIED are bright on dark


def _swap_hw(a: np.ndarray) -> np.ndarray:
    """Swap H and W axes only, preserving channel axis if present."""
    if a.ndim == 2:
        return a.T
    return np.transpose(a, (1, 0, 2))


D4_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "r90":    lambda a: np.rot90(a, k=1),
    "r180":   lambda a: np.rot90(a, k=2),
    "r270":   lambda a: np.rot90(a, k=3),
    "flipH":  lambda a: np.fliplr(a),
    "flipV":  lambda a: np.flipud(a),
    "trans":  _swap_hw,
    "atrans": lambda a: _swap_hw(np.rot90(a, k=2)),
}

ROT_ANGLES: list[int] = [a for a in range(15, 360, 15) if a % 90 != 0]
PORTRAIT_ANGLES: list[int] = [5, 355, 10, 350, 15, 345]


def rotate_nearest(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))
    M[0, 2] += (new_w / 2.0) - (w / 2.0)
    M[1, 2] += (new_h / 2.0) - (h / 2.0)
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BG_VALUE,
    )


def _build_full() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    ts: dict[str, Callable[[np.ndarray], np.ndarray]] = dict(D4_TRANSFORMS)
    for a in ROT_ANGLES:
        ts[f"r{a}"] = (lambda ang: (lambda img: rotate_nearest(img, ang)))(a)
    return ts


def _build_portrait() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    ts: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "flipH": lambda a: np.fliplr(a),
    }
    for a in PORTRAIT_ANGLES:
        ts[f"r{a}"] = (lambda ang: (lambda img: rotate_nearest(img, ang)))(a)
    return ts


FULL_TRANSFORMS = _build_full()
PORTRAIT_TRANSFORMS = _build_portrait()


def transforms_for(preset: str) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    if preset == "d4":
        return D4_TRANSFORMS
    if preset == "portrait":
        return PORTRAIT_TRANSFORMS
    return FULL_TRANSFORMS


def list_stems(d: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def _process_one(args_tuple) -> tuple[str, int, int, int, float]:
    src_path, out_dir, stem, overwrite, preset, color = args_tuple
    t0 = time.perf_counter()
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(src_path, flag)
    if img is None:
        return (stem, 0, 0, 1, (time.perf_counter() - t0) * 1000)
    transforms = transforms_for(preset)
    wrote = skipped = failed = 0
    for tag, fn in transforms.items():
        out_path = Path(out_dir) / f"{stem}__{tag}.png"
        if not overwrite and out_path.exists():
            skipped += 1
            continue
        arr = np.ascontiguousarray(fn(img))
        if not cv2.imwrite(str(out_path), arr, PNG_PARAMS):
            failed += 1
            continue
        wrote += 1
    return (stem, wrote, skipped, failed, (time.perf_counter() - t0) * 1000)


def _run_side(label: str, real_dir: Path, aug_dir: Path, overwrite: bool,
              jobs: int, preset: str, color: bool) -> tuple[int, int, int]:
    stems = list_stems(real_dir)
    if not stems:
        print(f"[{label}] no source files in {real_dir}", file=sys.stderr)
        return (0, 0, 1)
    aug_dir.mkdir(parents=True, exist_ok=True)
    n = len(stems)
    n_tx = len(transforms_for(preset))
    work = [(str(stems[s]), str(aug_dir), s, overwrite, preset, color)
            for s in sorted(stems)]

    wrote = skipped = failed = 0
    t0 = time.perf_counter()
    if jobs <= 1:
        for i, item in enumerate(work, 1):
            stem, w, s, f, dt = _process_one(item)
            wrote += w; skipped += s; failed += f
            print(f"[{label}][{i}/{n}] {stem}  wrote={w}/{n_tx}"
                  + (f" skipped={s}" if s else "")
                  + (f" failed={f}" if f else "")
                  + f"  [{dt:.0f} ms]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_process_one, item): i + 1
                    for i, item in enumerate(work)}
            done = 0
            for fut in as_completed(futs):
                done += 1
                stem, w, s, f, dt = fut.result()
                wrote += w; skipped += s; failed += f
                print(f"[{label}][{done}/{n}] {stem}  wrote={w}/{n_tx}"
                      + (f" skipped={s}" if s else "")
                      + (f" failed={f}" if f else "")
                      + f"  [{dt:.0f} ms]", flush=True)
    dt = time.perf_counter() - t0
    print(f"[{label}] Done. wrote={wrote}  skipped={skipped}  errors={failed}  "
          f"sources={n}  total={dt:.2f} s")
    return (wrote, skipped, failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=["full", "d4", "portrait"],
                    default="full",
                    help="'full' = 7 D4 + 20 15-deg rotations (27, default); "
                         "'d4' = 7 bit-exact D4; "
                         "'portrait' = flipH + +/-5/10/15 rotations (7)")
    ap.add_argument("--outlines", action="store_true",
                    help="augment only outlines (skip source)")
    ap.add_argument("--source-only", action="store_true",
                    help="augment only source (skip outlines)")
    ap.add_argument("--jobs", type=int,
                    default=min(8, os.cpu_count() or 1),
                    help="parallel worker processes (default min(8, ncpu))")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing files in aug/ (default: skip)")
    args = ap.parse_args()

    if args.outlines and args.source_only:
        ap.error("--outlines and --source-only are mutually exclusive")

    from tied.config import load_config
    cfg = load_config()

    do_source = not args.outlines
    do_outline = not args.source_only

    overall_fail = 0
    if do_source:
        color = (cfg.source == "rgb")
        if not cfg.train_source_real.is_dir():
            print(f"source real dir does not exist: {cfg.train_source_real}",
                  file=sys.stderr)
            return 1
        _, _, f = _run_side(
            "source", cfg.train_source_real, cfg.train_source_aug,
            args.overwrite, args.jobs, args.preset, color=color)
        overall_fail += f
    if do_outline:
        if not cfg.train_outline_real.is_dir():
            print(f"outline real dir does not exist: {cfg.train_outline_real}",
                  file=sys.stderr)
            return 1
        _, _, f = _run_side(
            "outline", cfg.train_outline_real, cfg.train_outline_aug,
            args.overwrite, args.jobs, args.preset, color=False)
        overall_fail += f

    return 0 if overall_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
