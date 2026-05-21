#!/usr/bin/env python3
"""Train TIED (TED model) on the configured dataset.

Reads ``tied.toml`` for dataset root, source channels, and outline
encoding.

Layout:

    <root>/train/source/{real,aug}/<stem>.<ext>
    <root>/train/outline/{real,aug}/<stem>.<ext>
    <root>/test/{source,outline}/<stem>.<ext>     # optional

Loss: TEED's combined bdcn_loss2 (on 4 heads) + cats_loss with a
radius=4 tolerance band on the final fused output.

Saves ``best.pt`` (lowest test loss), ``last.pt`` (last epoch), and
``log.json`` with per-epoch metrics into <out-dir>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tied.config import load_config
from tied.dataset import TiedDataset
from tied.loss import teed_total_loss
from tied.model import TED, count_parameters


def _epoch(model: TED, loader: DataLoader, device: str,
           opt: torch.optim.Optimizer | None,
           radius: int) -> dict:
    is_train = opt is not None
    model.train(is_train)
    loss_sum = 0.0
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["source"].to(device, non_blocking=True)
            y = batch["outline"].to(device, non_blocking=True)
            preds = model(x)
            loss = teed_total_loss(preds, y, radius=radius)
            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss.item())
            n += 1
    return {"loss": loss_sum / max(1, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to save checkpoints and log")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits", nargs="+", default=["real", "aug"],
                    choices=["real", "aug"], help="train splits to use")
    ap.add_argument("--crop-size", type=int, default=352,
                    help="random crop size for training (default 352); "
                         "0 disables cropping (full image, batch must be 1)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tolerance", type=int, default=None,
                    help="override [loss].tolerance from tied.toml — "
                         "radius (in pixels) of the cats_loss boundary "
                         "tolerance band")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue training from a .pt checkpoint")
    args = ap.parse_args()

    if args.crop_size == 0 and args.batch_size != 1:
        print("--crop-size=0 requires --batch-size=1; forcing batch_size=1",
              file=sys.stderr)
        args.batch_size = 1

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    tolerance = tolerance if tolerance is not None else cfg.tolerance
    tolerance = tolerance
    train_root = cfg.root / "train"
    test_root = cfg.root / "test"
    crop = args.crop_size if args.crop_size > 0 else None

    train_ds = TiedDataset(
        train_root, source_mode=cfg.source, outline_mode=cfg.outline,
        splits=tuple(args.splits), crop_size=crop)
    print(f"train: {len(train_ds)} samples  source={cfg.source}  "
          f"outline={cfg.outline}  crop={crop}  batch={args.batch_size}")

    test_ds = None
    if (test_root / "source").is_dir() and (test_root / "outline").is_dir():
        try:
            test_ds = TiedDataset(
                test_root, source_mode=cfg.source, outline_mode=cfg.outline,
                flat=True)
            print(f"test:  {len(test_ds)} samples")
        except RuntimeError as e:
            print(f"test:  skipped ({e})")
    else:
        print("test:  not found, skipping evaluation")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0))
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda")) if test_ds is not None else None

    model = TED(in_channels=cfg.in_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"model: TED in_ch={cfg.in_channels}  "
          f"params={count_parameters(model):,}  device={args.device}  "
          f"lr={args.lr}  tolerance={tolerance}")

    start_epoch = 1
    best_metric = float("inf")
    log: list[dict] = []
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"--resume: not a file: {args.resume}", file=sys.stderr)
            return 1
        ckpt = torch.load(args.resume, map_location=args.device,
                          weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict):
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            if "best_loss" in ckpt:
                best_metric = float(ckpt["best_loss"])
        print(f"resumed: {args.resume}  start_epoch={start_epoch}")
        existing_log = args.out_dir / "log.json"
        if existing_log.is_file():
            try:
                log = json.loads(existing_log.read_text())
            except Exception as e:
                print(f"could not read {existing_log}: {e}", file=sys.stderr)

    t_start = time.perf_counter()
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            t0 = time.perf_counter()
            tr = _epoch(model, train_loader, args.device, opt, tolerance)
            ev = (_epoch(model, test_loader, args.device, None, tolerance)
                  if test_loader is not None else None)
            dt = time.perf_counter() - t0

            ref = ev if ev is not None else tr
            improved = ref["loss"] < best_metric

            line = (f"ep {epoch:4d}/{start_epoch + args.epochs - 1}  "
                    f"train loss={tr['loss']:.5f}")
            if ev is not None:
                line += f"  test loss={ev['loss']:.5f}"
            line += f"  [{dt:.1f}s]"
            if improved:
                line += "  <"
            print(line, flush=True)
            log.append({"epoch": epoch, "dt_s": dt, "train": tr, "test": ev})

            if improved:
                best_metric = ref["loss"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_metric,
                    "source": cfg.source,
                    "outline": cfg.outline,
                    "in_channels": cfg.in_channels,
                    "args": vars(args),
                }, args.out_dir / "best.pt")
            if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1:
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_metric,
                    "source": cfg.source,
                    "outline": cfg.outline,
                    "in_channels": cfg.in_channels,
                    "args": vars(args),
                }, args.out_dir / "last.pt")
    except KeyboardInterrupt:
        print("\ninterrupted — saving last.pt", flush=True)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_loss": best_metric,
            "source": cfg.source,
            "outline": cfg.outline,
            "in_channels": cfg.in_channels,
            "args": vars(args),
        }, args.out_dir / "last.pt")
    finally:
        (args.out_dir / "log.json").write_text(json.dumps(log, indent=2))
        print(f"done. total={time.perf_counter() - t_start:.1f}s  "
              f"log={args.out_dir/'log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
