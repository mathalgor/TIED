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
| `--no-aug`     | off        | shortcut for `--splits real` — skip the augmented set for quick iteration |
| `--tolerance`  | (from toml)| spatial tolerance in pixels. `teed`: radius of the `cats_loss` bdr/texture neighbourhood. `soft_bce`: target is max-pooled by 2r+1 before BCE — predictions within `r` px of a true edge are not punished as FPs (lines get up to `r` px thicker). `soft_jaccard`: ignored. |
| `--loss`       | `auto`     | `auto`, `teed`, `soft_jaccard`, `soft_bce`. See the table above. |
| `--hard-threshold` | 0.5    | binarisation threshold for the hard metric. Lower (e.g. 0.2) when using a tonal loss whose edge predictions sit well below 0.5. |
| `--model`      | `ted`      | `ted` (~60k params, default), `tedup` (~180k, wider + deeper dense block), `teddeep` (adds a 4th encoder stage, doubles the receptive field; returns 5 heads). Saved as `model_kind` in the checkpoint, so `tied-infer` reconstructs the right class automatically. |
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

Loss is selectable via `--loss` and auto-routed by default:

| kind            | who uses it                  | tonal output? | --tolerance? | notes |
|---|---|---|---|---|
| `teed`          | default for `outline=mono`   | no (binary GT)| yes (`cats_loss` radius) | `bdcn_loss2` on all 4 heads + `cats_loss` on fused |
| `soft_jaccard`  | default for `outline=gray`   | no (saturates)| ignored      | `1 - p·t / (p+t-p·t)` on fused head; sharpest edges but pushes p→1 anywhere t>0 |
| `soft_bce`      | opt-in for tonal gray output | **yes**       | **yes** (max-pools target by 2r+1; tonal-preserving) | class-balanced BCE-with-logits (`pos_weight = sum(1-t)/sum(t)`); optimum `p = t` pixel-wise; at `tolerance=0` it is strictly per-pixel |

Auto-routing:
- `outline = "mono"` → `teed` (binary target, TEED loss assumes binary GT).
- `outline = "gray"` → `soft_jaccard` (no binarisation; faint edges
  contribute proportionally). Override with `--loss soft_bce` or
  `--loss soft_mse` when you want the output to preserve target
  intensity instead of saturating to 0/1.

Why TEED loss is not auto-picked for `outline=gray`: both `bdcn_loss2`
and `cats_loss` implicitly binarise the target (`mask > 0`,
`label != 0`) when computing class balance and the texture/border
terms, which throws away the soft signal that `outline=gray` carries.

The chosen loss is stored in the checkpoint as `loss_kind` and printed
in the startup line.

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

- Three model variants are available, picked with `tied-train --model`:
  `ted` (~60k, default), `tedup` (~180k, wider + deeper dense block),
  `teddeep` (adds a 4th encoder stage at stride 8, doubles the
  receptive field, returns 5 heads). All accept the configured
  `in_channels` (3 for `source=rgb`, 1 for `source=gray`). The chosen
  variant is recorded in the checkpoint, so `tied-infer` reconstructs
  the correct class without any flag.
- TEED loss (`bdcn_loss2 + cats_loss`) auto-adapts its per-head weights
  to the head count: TEED-original `(1.1, 0.7, 1.1, 1.3)` for 4-head
  models, uniform `1.0` for `teddeep`'s 5.
