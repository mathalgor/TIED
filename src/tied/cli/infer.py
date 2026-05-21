#!/usr/bin/env python3
"""Run a trained TIED checkpoint on a folder of images.

Reads source channels and outline encoding from ``tied.toml``. The
input folder is mandatory — pass it via ``--input``. Output PNGs are
written into ``--out-dir``.

Outputs PNGs sized to match the input (after resize-to-multiple-of-8
inside the model). Output is bright-on-dark, matching the TIED outline
convention:
  * ``--mode mono``  — threshold the sigmoid at ``--threshold`` (default
    0.5), write 0/255.
  * ``--mode gray``  — write the sigmoid scaled to [0, 255].
  * ``--mode auto``  — pick from the checkpoint's outline mode.
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
from tied.model import TED


def _load_checkpoint(path: Path, device: str) -> tuple[TED, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    in_ch = int(ckpt.get("in_channels", 3))
    model = TED(in_channels=in_ch).to(device)
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
    ap.add_argument("--mode", choices=["auto", "mono", "gray"],
                    default="auto",
                    help="output encoding (default: from checkpoint)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="(mono) sigmoid threshold (default 0.5)")
    args = ap.parse_args()

    cfg = load_config()
    if not args.ckpt.is_file():
        print(f"--ckpt not a file: {args.ckpt}", file=sys.stderr)
        return 1

    model, ckpt = _load_checkpoint(args.ckpt, args.device)
    source_mode = str(ckpt.get("source", cfg.source))
    outline_mode = str(ckpt.get("outline", cfg.outline))
    mode = outline_mode if args.mode == "auto" else args.mode

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

    print(f"ckpt={args.ckpt}  source={source_mode}  outline-mode={mode}  "
          f"files={len(files)}  device={args.device}")
    t0 = time.perf_counter()
    for i, p in enumerate(files, 1):
        img = _read_source(p, source_mode)              # (H, W, C) uint8
        h, w = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(args.device)
        x_in = TED.resize_input(x)
        with torch.no_grad():
            preds = model(x_in)
        fused = preds[-1]
        if fused.shape[-2:] != (h, w):
            fused = F.interpolate(fused, size=(h, w),
                                  mode="bicubic", align_corners=False)
        prob = torch.sigmoid(fused).squeeze().cpu().numpy()
        if mode == "mono":
            out = ((prob >= args.threshold).astype(np.uint8)) * 255
        else:
            out = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        out_path = args.out_dir / f"{p.stem}.png"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_PNG_COMPRESSION, 7])
        print(f"[{i}/{len(files)}] {p.name} -> {out_path.name}", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
