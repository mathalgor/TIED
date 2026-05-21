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
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tied.config import load_config
from tied.dataset import TiedDataset
from tied.loss import LOSS_KINDS, compute_loss, hard_pixel_counts, resolve_loss
from tied.model import TED, count_parameters


def _next_patience(p: int) -> int:
    """Grow patience along 2,3,4,6,8,12,16,24,32,48,64,...
    Alternates x1.5 (from a power of two) and x4/3 (to the next power
    of two). Slower than plain doubling so early plateaux still get
    quick retries while late ones earn long pauses."""
    is_pot = p > 0 and (p & (p - 1)) == 0
    return int(round(p * 1.5)) if is_pot else int(round(p * 4 / 3))


def _epoch(model: TED, loader: DataLoader, device: str,
           opt: torch.optim.Optimizer | None,
           radius: int, loss_kind: str,
           hard_threshold: float = 0.5) -> dict:
    is_train = opt is not None
    model.train(is_train)
    loss_sum = 0.0
    n = 0
    n_wrong = n_union = n_total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["source"].to(device, non_blocking=True)
            y = batch["outline"].to(device, non_blocking=True)
            preds = model(x)
            loss = compute_loss(loss_kind, preds, y, radius=radius)
            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss.item())
            n += 1
            hc = hard_pixel_counts(preds[-1], y, threshold=hard_threshold)
            n_wrong += hc["wrong_px"]
            n_union += hc["union_px"]
            n_total += hc["total_px"]
    hard = (n_wrong / n_union) if n_union > 0 else 0.0
    return {
        "loss": loss_sum / max(1, n),
        "hard": hard,
        "wrong_px": n_wrong,
        "union_px": n_union,
        "total_px": n_total,
    }


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
    ap.add_argument("--hard-threshold", type=float, default=0.5,
                    help="binarisation threshold for the hard metric "
                         "(default 0.5). Lower it (e.g. 0.2) for tonal "
                         "losses where edge predictions sit well below "
                         "0.5 and would otherwise all be counted wrong.")
    ap.add_argument("--loss", choices=("auto",) + LOSS_KINDS,
                    default="auto",
                    help="loss function. 'auto' (default) picks 'teed' "
                         "for outline=mono and 'soft_jaccard' for "
                         "outline=gray. Tonal alternatives that preserve "
                         "intensity in gray outputs: 'soft_bce' (BCE "
                         "with float targets, optimum p=t) and "
                         "'soft_mse'. 'teed' uses all 4 multi-scale "
                         "heads with cats_loss tolerance; the soft "
                         "losses operate on the fused output only and "
                         "ignore --tolerance.")
    ap.add_argument("--best-metric", choices=["auto", "loss", "hard"],
                    default="auto",
                    help="criterion for best.pt: 'loss' (TEED total) or "
                         "'hard' (MCED-style wrong/union after "
                         "binarising at 0.5). 'auto' (default): hard "
                         "for outline=mono, loss for outline=gray.")
    ap.add_argument("--rollback-on-plateau", action="store_true",
                    help="when training stops improving, reload the in-"
                         "memory best snapshot, re-init Adam, bump the "
                         "RNG seed and try again. Patience starts at "
                         "--initial-patience and grows along "
                         "2,3,4,6,8,12,16,24,32,48,... on every "
                         "consecutive failed rollback. A real "
                         "improvement resets patience.")
    ap.add_argument("--initial-patience", type=int, default=4,
                    help="(--rollback-on-plateau) epochs without "
                         "improvement before the first rollback (default 4)")
    ap.add_argument("--max-rollbacks", type=int, default=20,
                    help="(--rollback-on-plateau) hard cap on rollbacks "
                         "per run (default 20)")
    ap.add_argument("--lr-adapt", action="store_true",
                    help="adapt LR per epoch: on no improvement multiply "
                         "by --lr-shrink, on improvement multiply by "
                         "--lr-grow. Capped at --lr (top) and --lr-min "
                         "(bottom). Rollback (if enabled) resets LR to "
                         "the initial value.")
    ap.add_argument("--lr-shrink", type=float, default=0.9,
                    help="(--lr-adapt) per-epoch shrink factor on no "
                         "improvement (default 0.9)")
    ap.add_argument("--lr-grow", type=float, default=1.2,
                    help="(--lr-adapt) per-epoch grow factor on "
                         "improvement (default 1.2)")
    ap.add_argument("--lr-min", type=float, default=None,
                    help="(--lr-adapt) lower bound for LR (default lr*0.01)")
    args = ap.parse_args()

    if args.crop_size == 0 and args.batch_size != 1:
        print("--crop-size=0 requires --batch-size=1; forcing batch_size=1",
              file=sys.stderr)
        args.batch_size = 1

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    tolerance = args.tolerance if args.tolerance is not None else cfg.tolerance
    args.tolerance = tolerance
    best_metric_name = args.best_metric
    if best_metric_name == "auto":
        best_metric_name = "hard" if cfg.outline == "mono" else "loss"
    args.best_metric = best_metric_name
    loss_kind = resolve_loss(args.loss, cfg.outline)
    args.loss = loss_kind
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
          f"lr={args.lr}  tolerance={tolerance}  "
          f"best_metric={best_metric_name}  loss={loss_kind}")

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
            # Prefer a metric matching the current criterion, then any
            # known fields (forward-compatible with older checkpoints).
            for key in (f"best_{best_metric_name}", "best_metric", "best_loss"):
                if key in ckpt:
                    best_metric = float(ckpt[key])
                    break
        print(f"resumed: {args.resume}  start_epoch={start_epoch}  "
              f"best_{best_metric_name}={best_metric:.6f}")
        existing_log = args.out_dir / "log.json"
        if existing_log.is_file():
            try:
                log = json.loads(existing_log.read_text())
            except Exception as e:
                print(f"could not read {existing_log}: {e}", file=sys.stderr)

    # In-memory snapshot of best weights (for rollback on plateau).
    # If we resumed from best.pt the current model IS that best, so
    # snapshot it now.
    best_state = (copy.deepcopy(model.state_dict())
                  if best_metric != float("inf") else None)
    # "alt" snapshot: best state OBSERVED since the last new best —
    # strictly worse than best, but typically from a different
    # trajectory after a rollback. Reset on every new best.
    alt_state = None
    alt_metric = float("inf")
    no_improve = 0
    rollback_count = 0          # consecutive failed rollbacks; resets on improvement
    total_rollbacks = 0         # monotonic, drives the RNG seed
    patience = max(1, int(args.initial_patience))

    # Per-epoch adaptive LR state.
    current_lr = args.lr
    lr_min = args.lr_min if args.lr_min is not None else args.lr * 0.01

    def _set_lr(lr: float) -> None:
        for pg in opt.param_groups:
            pg["lr"] = lr

    t_start = time.perf_counter()
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            t0 = time.perf_counter()
            tr = _epoch(model, train_loader, args.device, opt,
                        tolerance, loss_kind, args.hard_threshold)
            ev = (_epoch(model, test_loader, args.device, None,
                         tolerance, loss_kind, args.hard_threshold)
                  if test_loader is not None else None)
            dt = time.perf_counter() - t0

            ref = ev if ev is not None else tr
            current = ref[best_metric_name]
            improved = current < best_metric

            line = (f"ep {epoch:4d}/{start_epoch + args.epochs - 1}  "
                    f"train loss={tr['loss']:.5f} hard={tr['hard']:.4f}")
            if ev is not None:
                line += (f"  test loss={ev['loss']:.5f} "
                         f"hard={ev['hard']:.4f}")
            line += f"  [{dt:.1f}s]"
            if args.lr_adapt:
                line += f"  lr={current_lr:.2e}"
            if improved:
                line += "  <"
            print(line, flush=True)
            log.append({"epoch": epoch, "dt_s": dt, "train": tr, "test": ev})

            if improved:
                best_metric = current
                best_state = copy.deepcopy(model.state_dict())
                alt_state = None
                alt_metric = float("inf")
                no_improve = 0
                rollback_count = 0
                patience = max(1, int(args.initial_patience))
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                    f"best_{best_metric_name}": best_metric,
                    "source": cfg.source,
                    "outline": cfg.outline,
                    "in_channels": cfg.in_channels,
                    "loss_kind": loss_kind,
                    "args": vars(args),
                }, args.out_dir / "best.pt")
            else:
                no_improve += 1
                # Best NON-best state since the last improvement — used
                # as an alternate rollback target.
                if current == current and current < alt_metric:
                    alt_metric = current
                    alt_state = copy.deepcopy(model.state_dict())

            # Adaptive LR: bump on improvement (capped at initial),
            # shrink on no improvement (floored at lr_min).
            if args.lr_adapt:
                if improved:
                    new_lr = min(current_lr * args.lr_grow, args.lr)
                else:
                    new_lr = max(current_lr * args.lr_shrink, lr_min)
                if new_lr != current_lr:
                    current_lr = new_lr
                    _set_lr(current_lr)

            if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1:
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                    f"best_{best_metric_name}": best_metric,
                    "source": cfg.source,
                    "outline": cfg.outline,
                    "in_channels": cfg.in_channels,
                    "loss_kind": loss_kind,
                    "args": vars(args),
                }, args.out_dir / "last.pt")

            # Plateau check: roll back to best (or alt), re-init Adam,
            # bump RNG seed, grow patience for the next try.
            if (args.rollback_on_plateau
                    and best_state is not None
                    and no_improve >= patience
                    and rollback_count < args.max_rollbacks):
                rollback_count += 1
                total_rollbacks += 1
                use_alt = (alt_state is not None and rollback_count % 2 == 0)
                src_state = alt_state if use_alt else best_state
                src_label = "alt" if use_alt else "best"
                src_metric = alt_metric if use_alt else best_metric
                model.load_state_dict(src_state)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                current_lr = args.lr
                new_seed = args.seed + 1000 * total_rollbacks + 37
                torch.manual_seed(new_seed)
                no_improve = 0
                patience = max(patience + 1, _next_patience(patience))
                print(f"  [rollback #{rollback_count}] reloaded {src_label} "
                      f"({best_metric_name}={src_metric:.6f}), seed->"
                      f"{new_seed}, next patience={patience}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — saving last.pt", flush=True)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            f"best_{best_metric_name}": best_metric,
            "source": cfg.source,
            "outline": cfg.outline,
            "in_channels": cfg.in_channels,
            "loss_kind": loss_kind,
            "args": vars(args),
        }, args.out_dir / "last.pt")
    finally:
        (args.out_dir / "log.json").write_text(json.dumps(log, indent=2))
        print(f"done. total={time.perf_counter() - t_start:.1f}s  "
              f"log={args.out_dir/'log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
