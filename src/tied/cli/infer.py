#!/usr/bin/env python3
"""Run a trained TIED checkpoint on a folder of images.

Reads source channels from ``tied.toml`` / the checkpoint. The input
folder is mandatory — pass it via ``--input``. Output PNGs are written
into ``--out-dir``, sized to match each input.

The output is **inverted grayscale** (dark edges on a white background)
for both outline modes, regardless of how the model was trained. This
matches the convention of edge maps you would inspect on screen and
makes mono vs gray checkpoints visually comparable. No thresholding is
applied — write `1 - sigmoid(pred)` scaled to [0, 255].
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tied.config import load_config
from tied.dataset import IMAGE_EXTS, _read_source
from tied.model import MODELS


def _load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    in_ch = int(ckpt.get("in_channels", 3))
    model_kind = str(ckpt.get("model_kind", "ted"))
    if model_kind not in MODELS:
        raise RuntimeError(f"unknown model_kind in checkpoint: {model_kind!r}")
    model = MODELS[model_kind](in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="path to a .pt checkpoint (e.g. ckpt/best.pt)")
    ap.add_argument("--input", type=Path, required=True,
                    help="folder of images to infer (required)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write outline PNGs")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config()
    if not args.ckpt.is_file():
        print(f"--ckpt not a file: {args.ckpt}", file=sys.stderr)
        return 1

    model, ckpt = _load_checkpoint(args.ckpt, args.device)
    source_mode = str(ckpt.get("source", cfg.source))
    outline_mode = str(ckpt.get("outline", cfg.outline))
    model_kind = str(ckpt.get("model_kind", "ted"))

    input_dir = args.input
    if not input_dir.is_dir():
        print(f"--input dir does not exist: {input_dir}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(input_dir.iterdir())
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"no images in {input_dir}", file=sys.stderr)
        return 1

    print(f"ckpt={args.ckpt}  model={model_kind}  source={source_mode}  "
          f"trained_outline={outline_mode}  files={len(files)}  "
          f"device={args.device}  (output: inverted grayscale)")
    t0 = time.perf_counter()
    for i, p in enumerate(files, 1):
        img = _read_source(p, source_mode)              # (H, W, C) uint8
        h, w = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(args.device)
        x_in = model.resize_input(x)
        with torch.no_grad():
            preds = model(x_in)
        fused = preds[-1]
        if fused.shape[-2:] != (h, w):
            fused = F.interpolate(fused, size=(h, w),
                                  mode="bicubic", align_corners=False)
        prob = torch.sigmoid(fused).squeeze().cpu().numpy()
        # Invert so edges are dark on a white background — easier to
        # inspect and identical for mono- and gray-trained checkpoints.
        out = np.clip((1.0 - prob) * 255.0, 0, 255).astype(np.uint8)
        out_path = args.out_dir / f"{p.stem}.png"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_PNG_COMPRESSION, 7])
        print(f"[{i}/{len(files)}] {p.name} -> {out_path.name}", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
