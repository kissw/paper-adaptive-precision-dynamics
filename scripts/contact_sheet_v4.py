"""Contact sheet generator for v4 obstacle-avoidance dataset visibility audit.

Purpose
-------
Estimate the actual fraction of obstacle-visible frames in expert_data_v4.h5
by visual inspection. v4 stores entire successful episodes (including pre-
approach normal driving and post-pass normal driving), so this script samples
frames uniformly across each episode and arranges them as a grid for manual
counting.

Output
------
- Multiple PNG pages, each containing N episodes (rows) x M frames (cols).
- Each cell labels episode_id and frame_idx so you can record visible-frame
  counts per episode.

Usage
-----
    python scripts/contact_sheet_v4.py \
        --input data/expert_data_v4.h5 \
        --output_dir outputs/contact_sheet_v4 \
        --frames_per_episode 20 \
        --episodes_per_page 10

Inspection workflow
-------------------
1. Run the script -> get N pages.
2. For each cell, mark whether an obstacle (vehicle) is recognizable.
3. Tally visible / total -> empirical visibility ratio.
4. Compare against the theoretical upper bound (~5-15%).
"""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def denormalize(img_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 in [-0.5, 0.5] -> (H, W, 3) uint8."""
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_uint = np.clip((img_hwc + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return img_uint


def sample_frames_per_episode(
    episode_ids: np.ndarray, frames_per_episode: int
) -> dict[int, list[int]]:
    """Return {ep_id: [global_frame_idx, ...]} with uniform sampling."""
    unique_eps = np.unique(episode_ids)
    samples: dict[int, list[int]] = {}
    for ep in unique_eps:
        idxs = np.where(episode_ids == ep)[0]
        if len(idxs) == 0:
            continue
        # uniform indices over the episode length
        positions = np.linspace(0, len(idxs) - 1, frames_per_episode).astype(int)
        # dedupe (short episodes may collapse)
        positions = np.unique(positions)
        samples[int(ep)] = [int(idxs[p]) for p in positions]
    return samples


def get_font():
    """Best-effort font load; fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, 10)
    return ImageFont.load_default()


def build_page(
    images: h5py.Dataset,
    episode_block: list[tuple[int, list[int]]],
    frames_per_episode: int,
    cell_size: int = 64,
    pad: int = 4,
    label_h: int = 14,
    upscale: int = 2,
) -> Image.Image:
    """Build one page image: rows = episodes, cols = sampled frames."""
    cell_img = cell_size * upscale
    n_rows = len(episode_block)
    n_cols = frames_per_episode
    row_label_w = 60  # left strip for "Ep XX" label

    page_w = row_label_w + n_cols * (cell_img + pad) + pad
    page_h = pad + n_rows * (cell_img + label_h + pad)

    page = Image.new("RGB", (page_w, page_h), (24, 24, 24))
    draw = ImageDraw.Draw(page)
    font = get_font()

    for r, (ep_id, frame_idxs) in enumerate(episode_block):
        y0 = pad + r * (cell_img + label_h + pad)

        # Row label (episode id)
        draw.text(
            (4, y0 + cell_img // 2 - 6),
            f"Ep {ep_id:02d}",
            fill=(220, 220, 220),
            font=font,
        )

        for c in range(n_cols):
            x0 = row_label_w + c * (cell_img + pad)
            if c < len(frame_idxs):
                gidx = frame_idxs[c]
                arr = denormalize(images[gidx])  # (H, W, 3) uint8
                pil = Image.fromarray(arr).resize(
                    (cell_img, cell_img), Image.NEAREST
                )
                page.paste(pil, (x0, y0))
                # frame index label below the cell
                draw.text(
                    (x0 + 2, y0 + cell_img + 1),
                    f"f{gidx}",
                    fill=(180, 180, 180),
                    font=font,
                )
            else:
                # missing cell (short episode)
                draw.rectangle(
                    [x0, y0, x0 + cell_img, y0 + cell_img],
                    fill=(40, 40, 40),
                )
    return page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to v4 HDF5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_per_episode", type=int, default=20)
    parser.add_argument("--episodes_per_page", type=int, default=10)
    parser.add_argument(
        "--upscale", type=int, default=2,
        help="Per-cell upscale factor (64*upscale px)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    with h5py.File(args.input, "r") as f:
        episode_ids = f["episode_ids"][:]
        images = f["images"]  # keep as HDF5 dataset (lazy read)

        n_total = len(episode_ids)
        unique_eps = sorted(np.unique(episode_ids).tolist())
        print(f"  total frames : {n_total}")
        print(f"  episodes     : {len(unique_eps)}")
        print(f"  frames/ep avg: {n_total / max(len(unique_eps), 1):.0f}")

        samples = sample_frames_per_episode(
            episode_ids, args.frames_per_episode
        )

        # Episode-length histogram (sanity check)
        ep_lens = [int((episode_ids == ep).sum()) for ep in unique_eps]
        print(f"  ep len min/median/max: "
              f"{min(ep_lens)}/{int(np.median(ep_lens))}/{max(ep_lens)}")

        n_pages = math.ceil(len(unique_eps) / args.episodes_per_page)
        print(f"  -> generating {n_pages} pages")

        for p in range(n_pages):
            block_eps = unique_eps[
                p * args.episodes_per_page : (p + 1) * args.episodes_per_page
            ]
            block = [(ep, samples[ep]) for ep in block_eps]
            page = build_page(
                images=images,
                episode_block=block,
                frames_per_episode=args.frames_per_episode,
                upscale=args.upscale,
            )
            out_path = out_dir / f"contact_sheet_page_{p + 1:02d}.png"
            page.save(out_path)
            print(f"    saved {out_path}  "
                  f"(eps {block_eps[0]}..{block_eps[-1]})")

    # Audit tally template
    tally_path = out_dir / "visibility_tally_template.csv"
    with open(tally_path, "w") as f:
        f.write("episode_id,total_sampled,visible_count,visible_ratio,notes\n")
        for ep in unique_eps:
            n_sampled = len(samples.get(ep, []))
            f.write(f"{ep},{n_sampled},,,\n")
    print(f"\nTally template: {tally_path}")
    print("Fill 'visible_count' per row, then compute aggregate ratio.")


if __name__ == "__main__":
    main()