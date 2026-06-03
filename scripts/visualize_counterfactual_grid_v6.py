#!/usr/bin/env python3
"""Visualize counterfactual v6 branch grids.

This script reads a counterfactual v6 HDF5 file produced by
scripts/collect_counterfactual_data_v6.py and creates one grid image per anchor.

Grid layout
-----------
Rows:
    Branches from the same cf_anchor_id, sorted by cf_bin_id and cf_sampled_steer.

Columns:
    Anchor/context frame at cf_branch_step == 0, followed by selected branch steps.

Default columns:
    anchor, step 1, step 2, ..., step 10

Example
-------
uv run python scripts/visualize_counterfactual_grid_v6.py \
    --input data/counterfactual_v6_smoke.h5 \
    --output_dir outputs/cf_grid_smoke \
    --max_anchors 10 \
    --steps 1,2,3,4,5,6,7,8,9,10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_int_list(text: str) -> list[int]:
    """Parse comma-separated integer list."""
    if text.strip() == "":
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def to_uint8_chw(img: np.ndarray) -> np.ndarray:
    """Convert CHW or HWC image to uint8 HWC.

    Supports:
      - CHW float in [-0.5, 0.5]
      - CHW float in [0, 1]
      - HWC float in [-0.5, 0.5]
      - HWC uint8 in [0, 255]
    """
    arr = np.asarray(img)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={arr.shape}")

    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)

    arr = arr.astype(np.float32)

    if arr.max() <= 1.0 and arr.min() >= -0.6:
        # Most PAActInf files are normalized to [-0.5, 0.5].
        if arr.min() < 0.0:
            arr = arr + 0.5
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize_image(img: np.ndarray, tile_size: int) -> Image.Image:
    """Convert to PIL image and resize to square tile."""
    pil = Image.fromarray(to_uint8_chw(img))
    if pil.size != (tile_size, tile_size):
        pil = pil.resize((tile_size, tile_size), Image.BILINEAR)
    return pil


def get_font(size: int) -> ImageFont.ImageFont:
    """Return a usable font without depending on a specific system font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def safe_get(data: dict[str, np.ndarray], key: str, default, idx: int):
    """Read per-frame value if key exists, otherwise return default."""
    if key not in data:
        return default
    return data[key][idx]


def load_h5(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load root-level datasets and attrs from HDF5."""
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arrays[key] = f[key][:]
        attrs = dict(f.attrs)

    required = [
        "images",
        "states",
        "actions",
        "episode_ids",
        "cf_anchor_id",
        "cf_branch_id",
        "cf_branch_step",
        "cf_bin_id",
        "cf_sampled_steer",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"Missing required counterfactual keys: {missing}")

    return arrays, attrs


def select_anchor_ids(
    data: dict[str, np.ndarray],
    *,
    anchor_ids: list[int] | None,
    mode: str,
    max_anchors: int | None,
) -> list[int]:
    """Select anchor ids optionally filtered by task label."""
    all_anchor_ids = np.unique(data["cf_anchor_id"].astype(np.int64))

    selected: list[int] = []
    for anchor_id in all_anchor_ids:
        mask = data["cf_anchor_id"] == anchor_id
        labels = data["task_labels"][mask] if "task_labels" in data else np.zeros(mask.sum(), dtype=np.int8)
        majority_label = int(np.round(np.mean(labels))) if labels.size else 0

        if mode == "clean" and majority_label != 0:
            continue
        if mode == "obstacle" and majority_label != 1:
            continue
        selected.append(int(anchor_id))

    if anchor_ids is not None:
        allowed = set(anchor_ids)
        selected = [a for a in selected if a in allowed]

    selected = sorted(selected)
    if max_anchors is not None:
        selected = selected[:max_anchors]

    return selected


def branch_sort_key(data: dict[str, np.ndarray], branch_id: int) -> tuple[int, float, int]:
    """Sort branches by bin id, sampled steer, branch id."""
    idx = np.where(data["cf_branch_id"] == branch_id)[0]
    if len(idx) == 0:
        return (999, 999.0, int(branch_id))
    i = int(idx[0])
    bin_id = int(data["cf_bin_id"][i])
    steer = float(data["cf_sampled_steer"][i])
    return (bin_id, steer, int(branch_id))


def frame_index_for_branch_step(
    data: dict[str, np.ndarray],
    *,
    branch_id: int,
    step: int,
) -> int | None:
    """Return index for a branch and cf_branch_step."""
    idx = np.where(
        (data["cf_branch_id"] == branch_id)
        & (data["cf_branch_step"] == step)
    )[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def make_cell_label(
    data: dict[str, np.ndarray],
    *,
    branch_id: int,
    row_idx: int,
) -> str:
    """Create row label for one branch."""
    idx = np.where(data["cf_branch_id"] == branch_id)[0]
    if len(idx) == 0:
        return f"{row_idx:02d}"

    i = int(idx[0])
    bin_id = int(data["cf_bin_id"][i])
    steer = float(data["cf_sampled_steer"][i])

    inv10 = bool(safe_get(data, "cf_branch_lane_invasion_10", False, i))
    col10 = bool(safe_get(data, "cf_branch_collision_10", False, i))
    max_cte = float(safe_get(data, "cf_max_abs_cte_delta_10", np.nan, i))

    flags = []
    if inv10:
        flags.append("LI")
    if col10:
        flags.append("COL")
    flag_text = ",".join(flags) if flags else "OK"

    return f"r{row_idx:02d} b{bin_id} s={steer:+.3f} {flag_text} cte={max_cte:.2f}"


def maybe_draw_bbox(
    draw: ImageDraw.ImageDraw,
    data: dict[str, np.ndarray],
    idx: int,
    *,
    x0: int,
    y0: int,
    tile_size: int,
    line_width: int = 2,
) -> None:
    """Draw obstacle_bbox if available and finite.

    BBox is assumed to be [x1, y1, x2, y2] in image pixel coordinates.
    If the source bbox convention is wrong, keep --draw_bbox disabled.
    """
    if "obstacle_bbox" not in data:
        return

    bbox = np.asarray(data["obstacle_bbox"][idx], dtype=np.float32)
    if bbox.shape[0] != 4 or not np.all(np.isfinite(bbox)):
        return

    x1, y1, x2, y2 = bbox.tolist()
    scale = tile_size / 64.0
    rect = [
        x0 + int(round(x1 * scale)),
        y0 + int(round(y1 * scale)),
        x0 + int(round(x2 * scale)),
        y0 + int(round(y2 * scale)),
    ]

    for k in range(line_width):
        draw.rectangle(
            [rect[0] - k, rect[1] - k, rect[2] + k, rect[3] + k],
            outline=(255, 255, 255),
        )


def make_grid_for_anchor(
    data: dict[str, np.ndarray],
    *,
    anchor_id: int,
    output_path: Path,
    steps: list[int],
    tile_size: int,
    label_width: int,
    header_height: int,
    row_gap: int,
    col_gap: int,
    draw_bbox: bool,
) -> dict[str, float | int]:
    """Create one grid PNG for one anchor."""
    anchor_mask = data["cf_anchor_id"] == anchor_id
    branch_ids = np.unique(data["cf_branch_id"][anchor_mask].astype(np.int64))
    branch_ids = sorted([int(b) for b in branch_ids], key=lambda b: branch_sort_key(data, b))

    if not branch_ids:
        raise ValueError(f"No branches found for anchor_id={anchor_id}")

    columns = [0] + steps
    n_rows = len(branch_ids)
    n_cols = len(columns)

    width = label_width + n_cols * tile_size + (n_cols - 1) * col_gap
    height = header_height + n_rows * tile_size + (n_rows - 1) * row_gap

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_small = get_font(11)
    font_header = get_font(13)

    labels = data["task_labels"][anchor_mask] if "task_labels" in data else np.zeros(anchor_mask.sum(), dtype=np.int8)
    task_label = int(np.round(np.mean(labels))) if labels.size else 0
    mode_text = "clean" if task_label == 0 else "obstacle"

    title = f"anchor={anchor_id} mode={mode_text} branches={n_rows}"
    draw.text((8, 4), title, fill=(0, 0, 0), font=font_header)

    for c, step in enumerate(columns):
        x = label_width + c * (tile_size + col_gap)
        text = "anchor" if step == 0 else f"+{step}"
        draw.text((x + 4, header_height - 18), text, fill=(0, 0, 0), font=font_small)

    branch_stats = []

    for r, branch_id in enumerate(branch_ids):
        y = header_height + r * (tile_size + row_gap)
        label = make_cell_label(data, branch_id=branch_id, row_idx=r)
        draw.text((6, y + 4), label, fill=(0, 0, 0), font=font_small)

        first_idx = np.where(data["cf_branch_id"] == branch_id)[0][0]
        branch_stats.append({
            "anchor_id": int(anchor_id),
            "branch_id": int(branch_id),
            "bin_id": int(data["cf_bin_id"][first_idx]),
            "steer": float(data["cf_sampled_steer"][first_idx]),
            "lane_invasion_10": int(bool(safe_get(data, "cf_branch_lane_invasion_10", False, first_idx))),
            "collision_10": int(bool(safe_get(data, "cf_branch_collision_10", False, first_idx))),
            "max_abs_cte_delta_10": float(safe_get(data, "cf_max_abs_cte_delta_10", np.nan, first_idx)),
            "max_abs_cte_delta_diag": float(safe_get(data, "cf_max_abs_cte_delta_diag", np.nan, first_idx)),
        })

        for c, step in enumerate(columns):
            idx = frame_index_for_branch_step(data, branch_id=branch_id, step=step)
            x = label_width + c * (tile_size + col_gap)

            if idx is None:
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(210, 210, 210))
                draw.text((x + 4, y + 4), "missing", fill=(0, 0, 0), font=font_small)
                continue

            tile = resize_image(data["images"][idx], tile_size=tile_size)
            canvas.paste(tile, (x, y))

            if draw_bbox:
                maybe_draw_bbox(draw, data, idx, x0=x, y0=y, tile_size=tile_size)

            step_inv = bool(safe_get(data, "cf_step_lane_invasion", False, idx))
            step_col = bool(safe_get(data, "cf_step_collision", False, idx))
            if step_inv or step_col:
                marker = "COL" if step_col else "LI"
                draw.rectangle([x, y, x + tile_size, y + 13], fill=(255, 255, 255))
                draw.text((x + 2, y + 1), marker, fill=(0, 0, 0), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    lane_inv_rate = float(np.mean([s["lane_invasion_10"] for s in branch_stats]))
    collision_rate = float(np.mean([s["collision_10"] for s in branch_stats]))
    mean_cte10 = float(np.mean([s["max_abs_cte_delta_10"] for s in branch_stats]))

    return {
        "anchor_id": int(anchor_id),
        "task_label": int(task_label),
        "n_branches": int(n_rows),
        "lane_invasion_rate_10": lane_inv_rate,
        "collision_rate_10": collision_rate,
        "mean_max_abs_cte_delta_10": mean_cte10,
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write anchor summary CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize counterfactual v6 branch grids.")
    parser.add_argument("--input", required=True, help="Counterfactual v6 HDF5 file.")
    parser.add_argument("--output_dir", default="outputs/cf_grid", help="Directory for PNG grids.")
    parser.add_argument("--anchor_ids", default=None, help="Optional comma-separated anchor ids, e.g. 0,1,2.")
    parser.add_argument("--mode", choices=["all", "clean", "obstacle"], default="all")
    parser.add_argument("--max_anchors", type=int, default=None)
    parser.add_argument("--steps", default="1,2,3,4,5,6,7,8,9,10", help="Branch steps to show.")
    parser.add_argument("--tile_size", type=int, default=96)
    parser.add_argument("--label_width", type=int, default=260)
    parser.add_argument("--header_height", type=int, default=48)
    parser.add_argument("--row_gap", type=int, default=3)
    parser.add_argument("--col_gap", type=int, default=3)
    parser.add_argument("--draw_bbox", action="store_true", help="Draw obstacle_bbox if available.")
    args = parser.parse_args()

    data, attrs = load_h5(args.input)
    output_dir = Path(args.output_dir)
    steps = parse_int_list(args.steps)
    anchor_ids = parse_int_list(args.anchor_ids) if args.anchor_ids is not None else None

    selected = select_anchor_ids(
        data,
        anchor_ids=anchor_ids,
        mode=args.mode,
        max_anchors=args.max_anchors,
    )

    if not selected:
        raise RuntimeError("No anchors selected. Check --anchor_ids or --mode.")

    print("=" * 100)
    print(f"input: {args.input}")
    print(f"output_dir: {output_dir}")
    print(f"selected anchors: {selected}")
    print(f"steps: {steps}")
    print(f"draw_bbox: {args.draw_bbox}")
    if attrs:
        print("attrs:")
        for key in ["collection_type", "anchors_collected", "branches_collected", "context_len", "branch_horizon", "diagnostic_horizon"]:
            if key in attrs:
                print(f"  {key}: {attrs[key]}")

    summary_rows = []
    for anchor_id in selected:
        mask = data["cf_anchor_id"] == anchor_id
        labels = data["task_labels"][mask] if "task_labels" in data else np.zeros(mask.sum(), dtype=np.int8)
        task_label = int(np.round(np.mean(labels))) if labels.size else 0
        mode_name = "clean" if task_label == 0 else "obstacle"

        out_path = output_dir / f"anchor_{anchor_id:04d}_{mode_name}.png"
        row = make_grid_for_anchor(
            data,
            anchor_id=anchor_id,
            output_path=out_path,
            steps=steps,
            tile_size=args.tile_size,
            label_width=args.label_width,
            header_height=args.header_height,
            row_gap=args.row_gap,
            col_gap=args.col_gap,
            draw_bbox=args.draw_bbox,
        )
        row["path"] = str(out_path)
        summary_rows.append(row)
        print(
            f"saved {out_path} | "
            f"branches={row['n_branches']} "
            f"lane_inv10={row['lane_invasion_rate_10']:.3f} "
            f"col10={row['collision_rate_10']:.3f} "
            f"mean_cte10={row['mean_max_abs_cte_delta_10']:.3f}"
        )

    summary_path = output_dir / "anchor_grid_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print("=" * 100)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()