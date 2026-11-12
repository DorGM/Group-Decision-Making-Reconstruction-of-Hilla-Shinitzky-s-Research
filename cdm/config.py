"""Configuration: OQ switches and the data root (plan §8, §9.2)."""
import os
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = CODE_ROOT / "config" / "defaults.yaml"


def load_config(path=None):
    """Read the YAML config. Default file: config/defaults.yaml (plan §9.2).

    `path` is read INSTEAD of the default file; nothing is merged, so an
    alternative file must be complete.
    """
    with open(path or DEFAULTS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_switch(oq_id, config=None):
    """Return the default value of one open question, e.g. get_switch("OQ9").

    IDs are the verbatim plan §8 IDs. Unknown IDs raise KeyError.
    """
    config = config if config is not None else load_config()
    return config["open_questions"][oq_id]["default"]


def data_root(config=None):
    """Folder holding the raw data (plan §2.1.1 D0b; layout in docs/DATA.md).

    Order: env var CDM_DATA_ROOT, then config "data_root" (default `data`;
    relative paths are taken relative to the repository root).
    """
    env = os.environ.get("CDM_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    config = config if config is not None else load_config()
    root = Path(config["data_root"])
    if not root.is_absolute():
        root = CODE_ROOT / root
    return root.resolve()


def tie_abs_tol(config=None):
    """Absolute tolerance for ties between float scores (top-level key
    `tie_abs_tol`; amendments AM3/AM9). 0 = exact.

    Raises ValueError if the entry is missing or is not a number >= 0
    (PyYAML reads "1e-9" without a decimal point as text).
    """
    config = config if config is not None else load_config()
    if "tie_abs_tol" not in config:
        raise ValueError("config has no 'tie_abs_tol' entry (use 0 for exact comparison)")
    tol = config["tie_abs_tol"]
    if isinstance(tol, (bool, str)) or tol is None:
        raise ValueError(f"tie_abs_tol must be a number >= 0, got {tol!r}")
    tol = float(tol)
    if not (tol >= 0.0 and tol != float("inf")):
        raise ValueError(f"tie_abs_tol must be a finite number >= 0, got {tol!r}")
    return tol
