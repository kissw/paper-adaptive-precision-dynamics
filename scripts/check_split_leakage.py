#!/usr/bin/env python3
"""Verify leakage-free train/valid split for mixed v5+v6 WM training data.

Checks:
  1. v5 (expert) part: train/valid episode_id intersection is empty.
  2. v6 (CF) part:    train/valid cf_anchor_id intersection is empty.
  3. is_counterfactual ratio is similar between train and valid (distribution match).
  4. Every key in each merged file has length == total_frames (no padding errors).

Usage:
    uv run python scripts/check_split_leakage.py \\
        --train  data/wm_train_mixed_v6_train.h5 \\
        --valid  data/wm_train_mixed_v6_valid.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def load_keys(path: str | Path, keys: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                out[k] = f[k][:]
        out["_total_frames"] = np.array(f["images"].shape[0])
        out["_all_keys"] = list(f.keys())
        out["_shapes"] = {k: f[k].shape for k in f.keys()}
    return out


def check_key_lengths(data: dict, path: str) -> list[str]:
    """All datasets must have first dim == total_frames."""
    errors = []
    total = int(data["_total_frames"])
    for k, shape in data["_shapes"].items():
        if shape and shape[0] != total:
            errors.append(f"  KEY LENGTH MISMATCH: '{k}' shape={shape}, expected {total}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage check for train/valid merged HDF5")
    parser.add_argument("--train", required=True, help="Merged train HDF5")
    parser.add_argument("--valid", required=True, help="Merged valid HDF5")
    parser.add_argument(
        "--cf_ratio_tol", type=float, default=0.15,
        help="Allowed absolute difference in is_counterfactual ratio between train/valid.",
    )
    args = parser.parse_args()

    needed = ["episode_ids", "is_counterfactual", "cf_anchor_id"]
    tr = load_keys(args.train, needed)
    va = load_keys(args.valid, needed)

    passed = True
    print("=" * 70)
    print(f"Train: {args.train}")
    print(f"Valid: {args.valid}")
    print("=" * 70)

    # ── 1. Key length consistency ─────────────────────────────────────────
    print("\n[1] Key length consistency")
    for label, data, path in [("train", tr, args.train), ("valid", va, args.valid)]:
        errs = check_key_lengths(data, path)
        if errs:
            for e in errs:
                print(e)
            passed = False
        else:
            print(f"  {label}: all {len(data['_all_keys'])} keys have consistent length ✓")

    # ── 2. v5 episode_id leakage ──────────────────────────────────────────
    print("\n[2] v5 episode_id leakage")
    tr_is_cf = tr.get("is_counterfactual", np.zeros(int(tr["_total_frames"]), dtype=bool)).astype(bool)
    va_is_cf = va.get("is_counterfactual", np.zeros(int(va["_total_frames"]), dtype=bool)).astype(bool)

    tr_v5_eps = set(tr["episode_ids"][~tr_is_cf].tolist()) if "episode_ids" in tr else set()
    va_v5_eps = set(va["episode_ids"][~va_is_cf].tolist()) if "episode_ids" in va else set()
    v5_overlap = tr_v5_eps & va_v5_eps
    if v5_overlap:
        print(f"  FAIL: {len(v5_overlap)} shared v5 episode_ids: {sorted(v5_overlap)[:10]}")
        passed = False
    else:
        print(f"  train v5 episodes={len(tr_v5_eps)}, valid v5 episodes={len(va_v5_eps)}")
        print(f"  intersection=0 ✓")

    # ── 3. v6 cf_anchor_id leakage ────────────────────────────────────────
    print("\n[3] v6 cf_anchor_id leakage")
    if "cf_anchor_id" in tr and "cf_anchor_id" in va:
        tr_cf_aids = set(tr["cf_anchor_id"][tr_is_cf].tolist())
        va_cf_aids = set(va["cf_anchor_id"][va_is_cf].tolist())
        # Remove anchor_id==0 from expert padding (expert rows have cf_anchor_id=0 by zero-fill)
        # Only consider rows where is_counterfactual=True
        v6_overlap = tr_cf_aids & va_cf_aids
        if v6_overlap:
            print(f"  FAIL: {len(v6_overlap)} shared cf_anchor_ids: {sorted(v6_overlap)[:10]}")
            passed = False
        else:
            print(f"  train CF anchors={len(tr_cf_aids)}, valid CF anchors={len(va_cf_aids)}")
            print(f"  intersection=0 ✓")
    else:
        print("  SKIP: cf_anchor_id not present (expert-only data)")

    # ── 4. is_counterfactual ratio ────────────────────────────────────────
    print("\n[4] is_counterfactual ratio (distribution match)")
    tr_cf_ratio = float(tr_is_cf.mean())
    va_cf_ratio = float(va_is_cf.mean())
    diff = abs(tr_cf_ratio - va_cf_ratio)
    status = "✓" if diff <= args.cf_ratio_tol else "WARN"
    print(f"  train CF ratio: {tr_cf_ratio:.3f}  ({int(tr_is_cf.sum())}/{int(tr['_total_frames'])} frames)")
    print(f"  valid CF ratio: {va_cf_ratio:.3f}  ({int(va_is_cf.sum())}/{int(va['_total_frames'])} frames)")
    print(f"  |diff|={diff:.3f}  (tol={args.cf_ratio_tol})  {status}")
    if diff > args.cf_ratio_tol:
        print("  NOTE: ratio mismatch may cause train/valid distribution shift")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if passed:
        print("RESULT: PASS — no leakage detected")
    else:
        print("RESULT: FAIL — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
