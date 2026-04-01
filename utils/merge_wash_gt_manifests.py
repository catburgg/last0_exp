#!/usr/bin/env python3
"""
Merge manifest.shardXXofYY.json from parallel wash_cosmos_dit_gt_cache runs into manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--output_name", type=str, default="manifest.json")
    a = p.parse_args()
    root = Path(a.output_dir).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1
    files = sorted(root.glob("manifest.shard*of*.json"))
    if not files:
        print(f"error: no manifest.shard*of*.json under {root}", file=sys.stderr)
        return 1
    if a.num_shards is not None and len(files) != a.num_shards:
        print(f"warn: expected {a.num_shards} shard manifests, found {len(files)}", file=sys.stderr)
    base: Dict[str, Any] | None = None
    entries: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            m = json.load(f)
        if base is None:
            base = {k: v for k, v in m.items() if k != "entries"}
        entries.extend(m.get("entries", []))
    out: Dict[str, Any] = dict(base or {})
    out["num_samples"] = len(entries)
    out["entries"] = entries
    out["shard_id"] = None
    out["merged"] = True
    out["num_shards_used_for_merge"] = len(files)
    outp = root / a.output_name
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Merged {len(files)} shard manifests -> {outp} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
