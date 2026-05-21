"""Project configuration loaded from ``tied.toml``.

Schema:

    [dataset]
    root    = "/absolute/path/to/tied-data"
    source  = "rgb"     # or "gray"
    outline = "mono"    # or "gray"

    [loss]
    tolerance = 4       # cats_loss boundary band radius (px)

Layout expected under ``root``:

    train/source/{real,aug}/<stem>.<ext>
    train/outline/{real,aug}/<stem>.<ext>
    test/source/<stem>.<ext>
    test/outline/<stem>.<ext>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:                                       # pragma: no cover
    import tomli as tomllib                               # type: ignore[no-redef]

CONFIG_NAME = "tied.toml"

SOURCE_MODES = ("rgb", "gray")
OUTLINE_MODES = ("mono", "gray")


def _find_config(start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        p = d / CONFIG_NAME
        if p.is_file():
            return p
    return None


@dataclass(frozen=True)
class TiedConfig:
    root: Path
    source: str       # "rgb" | "gray"
    outline: str      # "mono" | "gray"
    tolerance: int    # cats_loss radius in pixels

    @property
    def in_channels(self) -> int:
        return 3 if self.source == "rgb" else 1

    @property
    def train_source_real(self) -> Path: return self.root / "train" / "source" / "real"
    @property
    def train_source_aug(self)  -> Path: return self.root / "train" / "source" / "aug"
    @property
    def train_outline_real(self)-> Path: return self.root / "train" / "outline" / "real"
    @property
    def train_outline_aug(self) -> Path: return self.root / "train" / "outline" / "aug"
    @property
    def test_source(self)       -> Path: return self.root / "test" / "source"
    @property
    def test_outline(self)      -> Path: return self.root / "test" / "outline"


def load_config() -> TiedConfig:
    cfg_path = _find_config()
    if cfg_path is None:
        sys.exit(
            f"error: {CONFIG_NAME} not found in cwd or any parent.\n"
            f"       Copy {CONFIG_NAME}.template to {CONFIG_NAME} and edit it.")
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    except OSError as e:
        sys.exit(f"error: cannot read {cfg_path}: {e}")
    try:
        ds = cfg["dataset"]
        root_str = ds["root"]
    except (KeyError, TypeError):
        sys.exit(f"error: {cfg_path} is missing [dataset].root")
    source = str(ds.get("source", "rgb")).lower()
    outline = str(ds.get("outline", "mono")).lower()
    if source not in SOURCE_MODES:
        sys.exit(f"error: [dataset].source must be one of {SOURCE_MODES}, got {source!r}")
    if outline not in OUTLINE_MODES:
        sys.exit(f"error: [dataset].outline must be one of {OUTLINE_MODES}, got {outline!r}")
    loss_section = cfg.get("loss", {}) if isinstance(cfg.get("loss", {}), dict) else {}
    try:
        tolerance = int(loss_section.get("tolerance", 4))
    except (TypeError, ValueError):
        sys.exit(f"error: [loss].tolerance in {cfg_path} must be an integer")
    if tolerance < 0:
        sys.exit(f"error: [loss].tolerance must be >= 0, got {tolerance}")
    root = Path(str(root_str)).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"error: [dataset].root in {cfg_path} is not a directory: {root}")
    return TiedConfig(root=root, source=source, outline=outline,
                      tolerance=tolerance)
