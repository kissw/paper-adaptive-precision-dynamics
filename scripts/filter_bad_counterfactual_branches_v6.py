#!/usr/bin/env python3
"""Filter abnormal counterfactual v6 branches from an HDF5 dataset.

Purpose
-------
Remove branch-level CARLA restore artifacts from a counterfactual v6 dataset.

A branch is considered abnormal when its action-response is too small compared
with local branches from the same anchor and nearby steering bins.

The filter is branch-level:
    if one cf_branch_id is abnormal, all frames belonging to that branch are removed.

This preserves sequence consistency:
    one branch episode = context_len + branch_horizon frames.

Default criterion
-----------------
For each branch b at horizon H:

    cte_delta     = cte_H - cte_0
    heading_delta = heading_H - heading_0
    response      = |heading_delta| + cte_weight * |cte_delta|

For each branch, compute a local reference response as the median response of
the same anchor and neighboring steering bins. The branch is removed when:

    |sampled_steer| >= min_abs_steer
    and response / local_reference < low_ratio

Default:
    min_abs_steer = 0.10
    cte_weight    = 0.25
    low_ratio     = 0.35
    neighbor_bins = 1

Example
-------
uv run python scripts/filter_bad_counterfactual_branches_v6.py \
    --input data/counterfactual_v6_35_15_main.h5 \
    --output data/counterfactual_v6_35_15_main_filtered.h5 \
    --report outputs/cf_bad_branch_report_35_15.csv \
    --horizon 15 \
    --min_abs_steer 0.10 \
    --low_ratio 0.35
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_KEYS = [
    "states",
    "cf_anchor_id",
    "cf_branch_id",
    "cf_branch_step",
    "cf_bin_id",
    "cf_sampled_steer",
]


def infer_horizon(f: h5py.File, user_horizon: int | None) -> int:
    """Infer branch horizon from user argument, attrs, or cf_branch_step."""
    if user_horizon is not None:
        return int(user_horizon)

    if "branch_horizon" in f.attrs:
        return int(f.attrs["branch_horizon"])

    step = np.asarray(f["cf_branch_step"][:])
    positive = step[step > 0]
    if positive.size == 0:
        raise ValueError("Cannot infer horizon: cf_branch_step has no positive branch steps.")
    return int(positive.max())


def validate_input(f: h5py.File) -> int:
    """Validate required keys and return number of frames."""
    for key in REQUIRED_KEYS:
        if key not in f:
            raise KeyError(f"Missing required key: {key}")

    if "images" not in f:
        raise KeyError("Missing required key: images")

    n = int(f["images"].shape[0])
    for key in REQUIRED_KEYS:
        if f[key].shape[0] != n:
            raise ValueError(f"Length mismatch for {key}: {f[key].shape[0]} != {n}")

    if f["states"].shape[1] < 4:
        raise ValueError(f"states must have at least 4 dims [speed, steer, heading, cte], got {f['states'].shape}")

    return n


def build_branch_records(f: h5py.File, horizon: int, cte_weight: float) -> list[dict[str, Any]]:
    """Build per-branch response records without loading large image tensors."""
    aid = np.asarray(f["cf_anchor_id"][:], dtype=np.int64)
    bid = np.asarray(f["cf_branch_id"][:], dtype=np.int64)
    step = np.asarray(f["cf_branch_step"][:], dtype=np.int32)
    states = np.asarray(f["states"][:], dtype=np.float32)
    steer = np.asarray(f["cf_sampled_steer"][:], dtype=np.float32)
    bin_id = np.asarray(f["cf_bin_id"][:], dtype=np.int32)

    if "task_labels" in f:
        task = np.asarray(f["task_labels"][:], dtype=np.int8)
    else:
        task = np.zeros((len(bid),), dtype=np.int8)

    branch_ids = np.unique(bid)
    records: list[dict[str, Any]] = []

    for branch_id in branch_ids:
        idx0 = np.where((bid == branch_id) & (step == 0))[0]
        idxH = np.where((bid == branch_id) & (step == horizon))[0]

        if len(idx0) != 1 or len(idxH) != 1:
            # Malformed branch; mark as invalid response and remove later.
            idx_any = np.where(bid == branch_id)[0]
            i = int(idx_any[0])
            records.append({
                "anchor": int(aid[i]),
                "branch": int(branch_id),
                "bin": int(bin_id[i]),
                "steer": float(steer[i]),
                "task": int(task[i]),
                "cte0": np.nan,
                "cteH": np.nan,
                "cte_delta": np.nan,
                "heading0": np.nan,
                "headingH": np.nan,
                "heading_delta": np.nan,
                "response": np.nan,
                "malformed": True,
                "n_step0": int(len(idx0)),
                "n_stepH": int(len(idxH)),
            })
            continue

        i0 = int(idx0[0])
        iH = int(idxH[0])

        cte0 = float(states[i0, 3])
        cteH = float(states[iH, 3])
        heading0 = float(states[i0, 2])
        headingH = float(states[iH, 2])

        cte_delta = cteH - cte0
        heading_delta = headingH - heading0
        response = abs(heading_delta) + cte_weight * abs(cte_delta)

        records.append({
            "anchor": int(aid[iH]),
            "branch": int(branch_id),
            "bin": int(bin_id[iH]),
            "steer": float(steer[iH]),
            "task": int(task[iH]),
            "cte0": cte0,
            "cteH": cteH,
            "cte_delta": cte_delta,
            "heading0": heading0,
            "headingH": headingH,
            "heading_delta": heading_delta,
            "response": float(response),
            "malformed": False,
            "n_step0": 1,
            "n_stepH": 1,
        })

    return records


def detect_bad_branches(
    records: list[dict[str, Any]],
    *,
    min_abs_steer: float,
    low_ratio: float,
    neighbor_bins: int,
    min_ref_response: float,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Detect branch ids with abnormally weak response."""
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)

    for r in records:
        if r["malformed"]:
            continue
        if not np.isfinite(r["response"]):
            continue
        groups[(int(r["anchor"]), int(r["bin"]))].append(float(r["response"]))

    median_by_anchor_bin = {
        key: float(np.median(values))
        for key, values in groups.items()
        if len(values) > 0
    }

    bad_branch_ids: set[int] = set()
    bad_rows: list[dict[str, Any]] = []

    for r in records:
        branch_id = int(r["branch"])

        if r["malformed"]:
            rr = dict(r)
            rr["reason"] = "malformed_missing_step0_or_stepH"
            rr["ref_response"] = np.nan
            rr["ratio"] = np.nan
            bad_branch_ids.add(branch_id)
            bad_rows.append(rr)
            continue

        if abs(float(r["steer"])) < min_abs_steer:
            continue

        candidate_refs = []
        anchor = int(r["anchor"])
        current_bin = int(r["bin"])
        for nb in range(current_bin - neighbor_bins, current_bin + neighbor_bins + 1):
            key = (anchor, nb)
            if key in median_by_anchor_bin:
                candidate_refs.append(median_by_anchor_bin[key])

        if not candidate_refs:
            continue

        ref = float(np.median(candidate_refs))
        if ref < min_ref_response:
            continue

        response = float(r["response"])
        ratio = response / ref

        if ratio < low_ratio:
            rr = dict(r)
            rr["reason"] = "weak_local_response"
            rr["ref_response"] = ref
            rr["ratio"] = ratio
            bad_branch_ids.add(branch_id)
            bad_rows.append(rr)

    bad_rows = sorted(
        bad_rows,
        key=lambda x: (
            float("inf") if not np.isfinite(x.get("ratio", np.nan)) else float(x["ratio"]),
            int(x["anchor"]),
            int(x["branch"]),
        ),
    )
    return bad_branch_ids, bad_rows


def summarize(records: list[dict[str, Any]], bad_branch_ids: set[int]) -> dict[str, Any]:
    """Create summary stats."""
    total = len(records)
    bad = len(bad_branch_ids)

    by_task = {}
    for task_label in sorted(set(int(r["task"]) for r in records)):
        total_t = sum(1 for r in records if int(r["task"]) == task_label)
        bad_t = sum(1 for r in records if int(r["task"]) == task_label and int(r["branch"]) in bad_branch_ids)
        by_task[task_label] = {
            "total": total_t,
            "bad": bad_t,
            "bad_ratio": bad_t / max(1, total_t),
        }

    return {
        "total_branches": total,
        "bad_branches": bad,
        "bad_ratio": bad / max(1, total),
        "by_task": by_task,
    }


def write_report(path: Path, bad_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write bad branch report CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "anchor",
        "branch",
        "task",
        "bin",
        "steer",
        "cte0",
        "cteH",
        "cte_delta",
        "heading0",
        "headingH",
        "heading_delta",
        "response",
        "ref_response",
        "ratio",
        "reason",
        "n_step0",
        "n_stepH",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in bad_rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    summary_path = path.with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"total_branches: {summary['total_branches']}\n")
        f.write(f"bad_branches: {summary['bad_branches']}\n")
        f.write(f"bad_ratio: {summary['bad_ratio']}\n")
        for task_label, stats in summary["by_task"].items():
            f.write(f"task_{task_label}_total: {stats['total']}\n")
            f.write(f"task_{task_label}_bad: {stats['bad']}\n")
            f.write(f"task_{task_label}_bad_ratio: {stats['bad_ratio']}\n")


def make_keep_mask(f: h5py.File, bad_branch_ids: set[int]) -> np.ndarray:
    """Return frame-level keep mask."""
    bid = np.asarray(f["cf_branch_id"][:], dtype=np.int64)
    if not bad_branch_ids:
        return np.ones((len(bid),), dtype=bool)
    bad_arr = np.array(sorted(bad_branch_ids), dtype=np.int64)
    return ~np.isin(bid, bad_arr)


def copy_filtered_h5(
    input_path: Path,
    output_path: Path,
    *,
    keep_mask: np.ndarray,
    bad_branch_ids: set[int],
    bad_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    params: dict[str, Any],
    compression: str | None,
    chunk_rows: int,
) -> None:
    """Copy HDF5 datasets while filtering frame-level datasets by keep_mask."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        n = int(src["images"].shape[0])
        keep_count = int(keep_mask.sum())
        kept_indices = np.where(keep_mask)[0]

        for key in src.keys():
            obj = src[key]
            if not isinstance(obj, h5py.Dataset):
                print(f"SKIP non-dataset key: {key}")
                continue

            if obj.shape and obj.shape[0] == n:
                out_shape = (keep_count,) + obj.shape[1:]
                kwargs = {}
                if compression is not None and obj.ndim >= 1 and obj.size > 0:
                    kwargs["compression"] = compression

                out_ds = dst.create_dataset(
                    key,
                    shape=out_shape,
                    dtype=obj.dtype,
                    **kwargs,
                )

                dst_pos = 0
                for start in range(0, n, chunk_rows):
                    end = min(n, start + chunk_rows)
                    local_mask = keep_mask[start:end]
                    n_local = int(local_mask.sum())
                    if n_local == 0:
                        continue
                    chunk = obj[start:end]
                    out_ds[dst_pos:dst_pos + n_local] = chunk[local_mask]
                    dst_pos += n_local

                if dst_pos != keep_count:
                    raise RuntimeError(f"Copy position mismatch for key={key}: {dst_pos} != {keep_count}")
            else:
                # Keep only scalar/non-frame datasets if any. Most PAActInf flat HDF5 files
                # do not contain these. Avoid adding non-frame report datasets because
                # split_v5_train_valid.py expects frame-level root datasets only.
                print(f"SKIP non-frame dataset key={key}, shape={obj.shape}")

        for key, value in src.attrs.items():
            try:
                dst.attrs[key] = value
            except TypeError:
                dst.attrs[key] = str(value)

        dst.attrs["filtered_from"] = str(input_path)
        dst.attrs["collection_type_original"] = src.attrs.get("collection_type", "unknown")
        dst.attrs["collection_type"] = "counterfactual_v6_filtered"
        dst.attrs["filter_method"] = "anchor_bin_local_response"
        dst.attrs["filter_horizon"] = int(params["horizon"])
        dst.attrs["filter_min_abs_steer"] = float(params["min_abs_steer"])
        dst.attrs["filter_low_ratio"] = float(params["low_ratio"])
        dst.attrs["filter_cte_weight"] = float(params["cte_weight"])
        dst.attrs["filter_neighbor_bins"] = int(params["neighbor_bins"])
        dst.attrs["filter_min_ref_response"] = float(params["min_ref_response"])
        dst.attrs["total_frames_before_filter"] = int(n)
        dst.attrs["total_frames_after_filter"] = int(keep_count)
        dst.attrs["removed_frames"] = int(n - keep_count)
        dst.attrs["total_branches_before_filter"] = int(summary["total_branches"])
        dst.attrs["removed_branches"] = int(summary["bad_branches"])
        dst.attrs["removed_branch_ratio"] = float(summary["bad_ratio"])
        dst.attrs["kept_branches"] = int(summary["total_branches"] - summary["bad_branches"])

        bad_ids = np.array(sorted(bad_branch_ids), dtype=np.int64)
        # 78 ids is small enough for attrs. If future runs remove many branches,
        # this may still be fine; otherwise use the CSV report.
        dst.attrs["removed_branch_ids"] = bad_ids

        for task_label, stats in summary["by_task"].items():
            prefix = f"filter_task_{task_label}"
            dst.attrs[f"{prefix}_total_branches"] = int(stats["total"])
            dst.attrs[f"{prefix}_removed_branches"] = int(stats["bad"])
            dst.attrs[f"{prefix}_removed_ratio"] = float(stats["bad_ratio"])

        # Add frame-level keep marker after filtering. This has length equal to
        # output frame count, so downstream flat-HDF5 splitters remain compatible.
        dst.create_dataset(
            "cf_branch_filter_keep",
            data=np.ones((keep_count,), dtype=bool),
        )


def print_summary(summary: dict[str, Any], bad_rows: list[dict[str, Any]], max_print: int) -> None:
    """Print summary to stdout."""
    print("=" * 100)
    print("Branch filter summary")
    print(f"total branches: {summary['total_branches']}")
    print(f"bad branches:   {summary['bad_branches']}")
    print(f"bad ratio:      {summary['bad_ratio']:.6f}")

    for task_label, stats in summary["by_task"].items():
        name = "clean" if int(task_label) == 0 else "obstacle" if int(task_label) == 1 else f"task_{task_label}"
        print()
        print(name)
        print(f"  total:     {stats['total']}")
        print(f"  bad:       {stats['bad']}")
        print(f"  bad ratio: {stats['bad_ratio']:.6f}")

    print()
    print(f"Top {min(max_print, len(bad_rows))} suspicious branches:")
    for r in bad_rows[:max_print]:
        ratio = r.get("ratio", np.nan)
        ref = r.get("ref_response", np.nan)
        print(
            f"anchor={int(r['anchor']):04d} "
            f"branch={int(r['branch']):05d} "
            f"task={int(r['task'])} "
            f"bin={int(r['bin'])} "
            f"steer={float(r['steer']):+.3f} "
            f"cte_delta={float(r['cte_delta']):+.3f} "
            f"heading_delta={float(r['heading_delta']):+.3f} "
            f"response={float(r['response']):.4f} "
            f"ref={float(ref):.4f} "
            f"ratio={float(ratio):.3f} "
            f"reason={r.get('reason', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter abnormal counterfactual v6 branches.")
    parser.add_argument("--input", required=True, help="Input counterfactual v6 HDF5.")
    parser.add_argument("--output", required=True, help="Output filtered HDF5.")
    parser.add_argument("--report", default=None, help="CSV report path for removed branches.")
    parser.add_argument("--horizon", type=int, default=None, help="Branch horizon to evaluate. Defaults to HDF5 attr branch_horizon.")
    parser.add_argument("--min_abs_steer", type=float, default=0.10)
    parser.add_argument("--low_ratio", type=float, default=0.35)
    parser.add_argument("--cte_weight", type=float, default=0.25)
    parser.add_argument("--neighbor_bins", type=int, default=1)
    parser.add_argument("--min_ref_response", type=float, default=1e-6)
    parser.add_argument("--compression", default=None, choices=[None, "gzip", "lzf"])
    parser.add_argument("--chunk_rows", type=int, default=4096)
    parser.add_argument("--dry_run", action="store_true", help="Only report bad branches; do not write output HDF5.")
    parser.add_argument("--max_print", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".bad_branches.csv")

    with h5py.File(input_path, "r") as f:
        n = validate_input(f)
        horizon = infer_horizon(f, args.horizon)

        print("=" * 100)
        print(f"input: {input_path}")
        print(f"output: {output_path}")
        print(f"frames: {n}")
        print(f"horizon used: {horizon}")
        print(f"min_abs_steer: {args.min_abs_steer}")
        print(f"low_ratio: {args.low_ratio}")
        print(f"cte_weight: {args.cte_weight}")
        print(f"neighbor_bins: {args.neighbor_bins}")

        records = build_branch_records(f, horizon=horizon, cte_weight=args.cte_weight)
        bad_branch_ids, bad_rows = detect_bad_branches(
            records,
            min_abs_steer=args.min_abs_steer,
            low_ratio=args.low_ratio,
            neighbor_bins=args.neighbor_bins,
            min_ref_response=args.min_ref_response,
        )
        summary = summarize(records, bad_branch_ids)
        print_summary(summary, bad_rows, args.max_print)

        write_report(report_path, bad_rows, summary)
        print()
        print(f"report: {report_path}")
        print(f"summary report: {report_path.with_suffix('.summary.txt')}")

        if args.dry_run:
            print("dry_run=True, output HDF5 was not written.")
            return

        keep_mask = make_keep_mask(f, bad_branch_ids)

    params = {
        "horizon": horizon,
        "min_abs_steer": args.min_abs_steer,
        "low_ratio": args.low_ratio,
        "cte_weight": args.cte_weight,
        "neighbor_bins": args.neighbor_bins,
        "min_ref_response": args.min_ref_response,
    }

    copy_filtered_h5(
        input_path=input_path,
        output_path=output_path,
        keep_mask=keep_mask,
        bad_branch_ids=bad_branch_ids,
        bad_rows=bad_rows,
        summary=summary,
        params=params,
        compression=args.compression,
        chunk_rows=args.chunk_rows,
    )

    print("=" * 100)
    print(f"wrote filtered HDF5: {output_path}")
    print(f"removed branches: {summary['bad_branches']}")
    print(f"removed frames: {summary['bad_branches']} * sequence_len")
    print("Done.")


if __name__ == "__main__":
    main()
