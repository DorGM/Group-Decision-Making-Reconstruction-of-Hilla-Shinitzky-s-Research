"""Guard of the rule-baseline check ("rule gate"): no model is trained or
evaluated on a real-data build unless the check is met on it (status
`passed`) or proceeding with the documented differences was authorised
(status `waived`).

Plan §2.1.1 Step D5: the rule-baseline check comes before any ML on the
real data. On every build some rule-baseline counts differ from the
thesis-reported counts by more than the tolerance
(docs/IMPLEMENTATION_DECISIONS.md G1). The project owner authorised
proceeding on the frozen builds `waiver_main` (primary) and
`waiver_politics_alt` (sensitivity) with the differences documented:
docs/RULE_BASELINE_CHECK.md, plan amendment AM17. The machine-readable
record is config/gate_status.yaml.

`require_gate(build_dir)` is called by every driver script in scripts/
before a build is loaded. It raises `GateError` unless ALL of these hold:
  * the folder holds a manifest whose `scenario` equals the folder name;
  * that name is listed in config/gate_status.yaml with status `passed`
    or `waived`;
  * the reference document of the record exists in the code tree;
  * the SHA-256 of the cases file equals both the hash in the manifest and
    the hash in the record (the build is frozen: the decision covers these
    bytes, not whatever is in the folder later).
"""
import hashlib
import json
from pathlib import Path

import yaml

from cdm.config import CODE_ROOT

GATE_STATUS_PATH = CODE_ROOT / "config" / "gate_status.yaml"
ALLOWED_STATUS = ("passed", "waived")
CASES_FILE = "worldA_cases.jsonl"
MANIFEST_FILE = "manifest.json"


class GateError(RuntimeError):
    """The build may not be used for ML (no `passed` / `waived` record)."""


def load_gate_status(path=None):
    """{build name: record} from config/gate_status.yaml."""
    with open(path or GATE_STATUS_PATH, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    builds = doc.get("builds") or {}
    if not isinstance(builds, dict):
        raise GateError("gate_status.yaml: `builds` must be a mapping")
    return builds


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_gate(build_dir, gate_status_path=None):
    """Check that the frozen build in `build_dir` may be used for ML.

    Returns a dict for the run record: build, status, role, date,
    decided_by, reference, amendment, cases_sha256. Raises GateError
    otherwise (see the module docstring for the conditions).
    `gate_status_path` is for the unit tests; the driver scripts always
    use config/gate_status.yaml.
    """
    build_dir = Path(build_dir)
    manifest_path = build_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise GateError(f"{build_dir} has no {MANIFEST_FILE}: not a frozen build")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    name = manifest.get("scenario")
    if not name or name != build_dir.name:
        raise GateError(f"manifest scenario {name!r} does not match the folder "
                        f"name {build_dir.name!r}")
    builds = load_gate_status(gate_status_path)
    if name not in builds:
        raise GateError(
            f"build {name!r} is not listed in {gate_status_path or GATE_STATUS_PATH}: "
            "it has no `passed` / `waived` record of the rule-baseline check, so no model may "
            f"be trained or evaluated on it. Listed builds: {sorted(builds)}")
    rec = dict(builds[name] or {})
    if rec.get("status") not in ALLOWED_STATUS:
        raise GateError(f"build {name!r} has gate status {rec.get('status')!r}; "
                        f"needed: one of {ALLOWED_STATUS}")
    for key in ("date", "reference", "cases_sha256"):
        if not rec.get(key):
            raise GateError(f"gate record of {name!r} has no `{key}`")
    if not (CODE_ROOT / rec["reference"]).is_file():
        raise GateError(f"gate record of {name!r}: reference {rec['reference']!r} "
                        "does not exist in the code tree")
    cases_path = build_dir / CASES_FILE
    if not cases_path.is_file():
        raise GateError(f"{build_dir} has no {CASES_FILE}")
    actual = _sha256(cases_path)
    in_manifest = (manifest.get("files") or {}).get(CASES_FILE)
    if actual != in_manifest:
        raise GateError(f"{cases_path} does not match the hash in its manifest")
    if actual != rec["cases_sha256"]:
        raise GateError(
            f"build {name!r}: the cases file ({actual[:12]}...) is not the frozen "
            f"file the gate record covers ({str(rec['cases_sha256'])[:12]}...)")
    out = {"build": name, "cases_sha256": actual}
    for key in ("status", "role", "date", "decided_by", "reference", "amendment"):
        out[key] = rec.get(key)
    out["date"] = str(out["date"])
    return out
