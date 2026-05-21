# TIED — TEED-Inspired Edge Detector

A small training pipeline for an edge / outline detector based on the
[TEED](https://github.com/xavysp/TEED) architecture (~58k params), with
an MCED-style dataset layout (separate `real/` and `aug/` subfolders)
and offline geometric augmentations. The model and loss start out
TEED-equivalent; a deeper MCED-style variant will be added later.

## 1. Install

```bash
git clone <this-repo> TIED
cd TIED
pip install -e .
```

Runtime dependencies: `numpy`, `opencv-python`, `torch>=2.0`. CUDA is
auto-detected (`--device cuda` is the default when available).

## 2. Configure

Copy the template and edit it:

```bash
cp tied.toml.template tied.toml
```

`tied.toml` is git-ignored — each developer keeps their own. All CLIs
walk up from the current working directory to find it.

```toml
[dataset]
root    = "/abs/path/to/your/tied-data"
source  = "rgb"     # or "gray" — picks the 3-ch vs 1-ch TED variant
outline = "mono"    # or "gray"
                    #   mono = white (255) on black (0); binarised at 128
                    #   gray = gray on black, kept as float in [0, 1]

[loss]
tolerance = 4       # cats_loss boundary band radius in pixels
                    # (predictions within this distance of a true edge
                    # are not punished). Override per-run with
                    # `tied-train --tolerance N`.
```

Note: TIED's outline convention is **bright edge on dark background**
(opposite of MCED, similar to TEED).

## 3. Dataset layout

```
<root>/
├── train/
│   ├── source/
│   │   ├── real/        # original images
│   │   └── aug/         # written by `tied-augment`
│   └── outline/
│       ├── real/        # ground-truth outlines for real/
│       └── aug/         # written by `tied-augment`
└── test/                # optional, used for eval
    ├── source/
    └── outline/
```

Pairing is by file stem (e.g. `cat.png` in `source/real/` pairs with
`cat.png` in `outline/real/`). Augmentation tags are appended after a
double underscore — `cat__r90.png`, `cat__flipH.png`, etc. — and must
match between source and outline.

## 4. CLIs

### `tied-augment` — offline geometric augmentations

Reads the dataset root from `tied.toml`. Same transform is applied to
source and outline, so pixels stay aligned. Background fill is `0`
(black, matching the bright-on-dark convention).

```bash
tied-augment                    # both sides, full preset (27 transforms)
tied-augment --preset portrait  # flipH + ±5/±10/±15° rotations (7 total)
tied-augment --preset d4        # 7 bit-exact dihedral transforms only
tied-augment --outlines         # only augment outlines (skip source)
tied-augment --source-only      # only augment source (skip outlines)
tied-augment --jobs 8 --overwrite
```

Presets:
- `full` (default) — 7 D4 transforms (`r90`, `r180`, `r270`, `flipH`,
  `flipV`, `trans`, `atrans`) plus 20 nearest-neighbour rotations at
  every 15° (skipping the D4 angles). 27 variants per input.
- `d4` — only the 7 bit-exact D4 transforms.
- `portrait` — `flipH` plus ±5°, ±10°, ±15° rotations. 7 variants. No
  vertical flip and no 90/180/270° rotations — suitable for upright
  faces.

Existing files are skipped unless `--overwrite` is passed.

### `tied-train` — training

```bash
tied-train --out-dir ckpt --epochs 50
```

Useful options:

| flag | default | meaning |
|---|---|---|
| `--out-dir`    | (required) | where to save `best.pt`, `last.pt`, `log.json` |
| `--epochs`     | 50         | number of epochs to run |
| `--lr`         | 1e-3       | Adam learning rate |
| `--batch-size` | 8          | train batch (eval is always 1) |
| `--crop-size`  | 352        | random crop size; `0` disables (full image, batch=1) |
| `--splits`     | real aug   | which `train/source/<split>` folders to use |
| `--tolerance`  | (from toml)| override `[loss].tolerance` for this run |
| `--best-metric`| `auto`     | `loss`, `hard`, or `auto` (= `hard` when outline=mono, `loss` when outline=gray). `hard` = MCED-style wrong/union after binarising sigmoid(pred) and target at 0.5 |
| `--rollback-on-plateau` | off | reload the in-memory best snapshot on a plateau, re-init Adam, bump RNG seed, retry |
| `--initial-patience` | 4 | (rollback) epochs without improvement before the first rollback; grows along 4,6,8,12,16,24,32,48,… on every failed rollback |
| `--max-rollbacks` | 20 | (rollback) hard cap on rollbacks per run |
| `--lr-adapt`   | off        | per-epoch LR shrink on no-improve, grow on improve (capped at `--lr` and floored at `--lr-min`) |
| `--lr-shrink`  | 0.9        | (lr-adapt) per-epoch shrink factor |
| `--lr-grow`    | 1.2        | (lr-adapt) per-epoch grow factor |
| `--lr-min`     | lr × 0.01  | (lr-adapt) lower bound for LR |
| `--device`     | auto       | `cuda` if available, else `cpu` |
| `--num-workers`| 4          | DataLoader workers |
| `--save-every` | 5          | overwrite `last.pt` every N epochs |
| `--resume`     | —          | continue from a `.pt` checkpoint |
| `--seed`       | 0          | RNG seed |

Loss: TEED's combined `bdcn_loss2` (on the 3 multi-scale heads + the
fused head) plus `cats_loss` (with the configured `tolerance` radius)
on the final fused output.

Per-epoch log line shows both `loss` and `hard` for train and (if
available) test, with a trailing `<` whenever the chosen `--best-metric`
improved on that epoch:

```
ep    7/50  train loss=2.31234 hard=0.0421  test loss=2.45123 hard=0.0398  [4.2s]  <
```

Outputs in `--out-dir` (only these three files — no per-epoch
checkpoints):
- `best.pt` — checkpoint with the lowest evaluation loss seen so far
  (falls back to train loss if `test/` is missing).
- `last.pt` — most recent checkpoint, rewritten every `--save-every`
  epochs and on Ctrl-C.
- `log.json` — per-epoch metrics.

Each checkpoint embeds `epoch`, `best_loss`, `source`, `outline`,
`in_channels`, and the full `args` namespace. Resuming reads everything
from the `.pt` itself — `log.json` is only re-loaded so new epochs get
appended.

```bash
# resume from best.pt or last.pt — start_epoch and best_loss are
# read from the checkpoint automatically. --epochs is the count of
# MORE epochs to run, not a new total.
tied-train --out-dir ckpt --epochs 20 --resume ckpt/best.pt
```

### `tied-infer` — inference

```bash
tied-infer --ckpt ckpt/best.pt --input data --out-dir outlines
```

Runs the checkpoint on every image in `--input` and writes one PNG per
image into `--out-dir` (sized to match each input). To infer on the
configured test split, pass it explicitly:

```bash
tied-infer --ckpt ckpt/best.pt --input "$ROOT/test/source" --out-dir outlines
```

Output is **inverted grayscale** — dark edges on a white background —
for both `outline=mono` and `outline=gray` checkpoints. No thresholding
is applied even for mono-trained models, so the raw confidence map is
visible and the two training modes are visually comparable.

Options:

| flag | default | meaning |
|---|---|---|
| `--ckpt`      | (required)  | path to a `.pt` checkpoint |
| `--input`     | (required)  | folder of images to run on |
| `--out-dir`   | (required)  | where to write outline PNGs |
| `--device`    | auto        | `cuda` or `cpu` |

## 5. Typical workflow

```bash
# 1. populate <root>/train/source/real and <root>/train/outline/real
# 2. generate aug/ for both sides
tied-augment --preset portrait

# 3. train
tied-train --out-dir ckpt --epochs 50

# 4. infer (point --input at any folder of images, e.g. the test split)
tied-infer --ckpt ckpt/best.pt --input "$ROOT/test/source" --out-dir outlines
```

## 6. Notes & roadmap

- The model is currently TEED's TED net with a configurable input
  channel count (3 for RGB sources, 1 for grayscale). The 1-channel
  variant is selected automatically when `[dataset].source = "gray"`.
- Loss carries TEED's tolerance band via `cats_loss(radius=...)`, so
  off-by-one predictions near true edges are not punished.
- Planned: a deeper, MCED-style model variant exposed via a
  `tied-train --model deep` flag (similar to MCED's `teed` / `teedup`
  switch), for cases where the small TED net saturates.
