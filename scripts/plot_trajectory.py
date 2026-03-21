"""Generate trajectory visualizations from JSONL evaluation logs.

Produces per-episode overview plots (speed/steering/EFE over time) and
map-view plots (XY colored by speed, acceleration, steering) suitable
for paper figures.

Usage:
    uv run python scripts/plot_trajectory.py --input_dir outputs/eval_v4/trajectories
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trajectory(jsonl_path):
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def plot_overview(frames, output_path, title=""):
    """4-panel overview: trajectory, speed+steer, EFE+epistemic, lateral dev."""
    x = [f["x"] for f in frames]
    y = [f["y"] for f in frames]
    speed = [f["speed_mps"] for f in frames]
    time = [f["timestamp"] for f in frames]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. XY trajectory colored by speed
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=speed, cmap="RdYlGn_r", s=3)
    ax.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax.plot(x[-1], y[-1], "rs", markersize=10, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory (color = speed m/s)")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Speed (m/s)")

    # 2. Speed and steering over time
    ax = axes[0, 1]
    ax.plot(time, speed, label="Speed (m/s)", color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2 = ax.twinx()
    steer = [f["action_steer"] for f in frames]
    ax2.plot(time, steer, label="Cmd Steer", color="red", alpha=0.7)
    ax2.set_ylabel("Commanded Steer", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.set_title("Speed & Steering Commands")

    # 3. EFE and Epistemic over time
    ax = axes[1, 0]
    efe = [f["efe_score"] for f in frames]
    epi = [f["epistemic_score"] for f in frames]
    ax.plot(time, efe, label="EFE", color="purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EFE Score", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax2 = ax.twinx()
    ax2.plot(time, epi, label="Epistemic", color="orange")
    ax2.set_ylabel("Epistemic Score", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax.set_title("Planning Scores Over Time")

    # 4. Lateral deviation + events
    ax = axes[1, 1]
    lat_dev = [f["lateral_dev"] for f in frames]
    ax.plot(time, lat_dev, label="Lateral Dev (m)", color="green")
    col_t = [f["timestamp"] for f in frames if f["collision"]]
    col_v = [f["lateral_dev"] for f in frames if f["collision"]]
    if col_t:
        ax.scatter(col_t, col_v, c="red", marker="x", s=50, label="Collision", zorder=5)
    inv_t = [f["timestamp"] for f in frames if f["lane_invasion"]]
    inv_v = [f["lateral_dev"] for f in frames if f["lane_invasion"]]
    if inv_t:
        ax.scatter(inv_t, inv_v, c="orange", marker=".", s=20, label="Lane Invasion", zorder=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateral Deviation (m)")
    ax.set_title("Lane Keeping Quality")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_map_view(frames, output_path, title=""):
    """3-panel map view: speed, acceleration, steering on XY plane."""
    x = np.array([f["x"] for f in frames])
    y = np.array([f["y"] for f in frames])
    speed = np.array([f["speed_mps"] for f in frames])
    yaw_rad = np.radians(np.array([f["yaw"] for f in frames]))
    action_steer = np.array([f["action_steer"] for f in frames])
    action_accel = np.array([f["action_accel"] for f in frames])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Heading arrows every N frames
    step = max(1, len(x) // 25)

    # 1. Speed map
    ax = axes[0]
    sc = ax.scatter(x, y, c=speed, cmap="RdYlGn_r", s=4)
    for i in range(0, len(x), step):
        dx = np.cos(yaw_rad[i]) * 3
        dy = np.sin(yaw_rad[i]) * 3
        ax.annotate(
            "",
            xy=(x[i] + dx, y[i] + dy),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
    ax.plot(x[0], y[0], "go", markersize=8)
    ax.plot(x[-1], y[-1], "rs", markersize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Speed (m/s)")
    plt.colorbar(sc, ax=ax)

    # 2. Acceleration map
    ax = axes[1]
    vmax_a = max(abs(action_accel.min()), abs(action_accel.max()), 0.5)
    sc = ax.scatter(x, y, c=action_accel, cmap="RdBu", s=4, vmin=-vmax_a, vmax=vmax_a)
    ax.plot(x[0], y[0], "go", markersize=8)
    ax.plot(x[-1], y[-1], "rs", markersize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Acceleration Command")
    plt.colorbar(sc, ax=ax)

    # 3. Steering map
    ax = axes[2]
    vmax_s = max(abs(action_steer.min()), abs(action_steer.max()), 0.5)
    sc = ax.scatter(x, y, c=action_steer, cmap="RdBu", s=4, vmin=-vmax_s, vmax=vmax_s)
    ax.plot(x[0], y[0], "go", markersize=8)
    ax.plot(x[-1], y[-1], "rs", markersize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.set_title("Steering Command")
    plt.colorbar(sc, ax=ax)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory plots from JSONL logs")
    parser.add_argument("--input_dir", required=True, help="Directory with trajectory_*.jsonl files")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/plots)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("trajectory_*.jsonl"))
    if not jsonl_files:
        print(f"No trajectory files found in {input_dir}")
        return

    for jsonl_path in jsonl_files:
        frames = load_trajectory(jsonl_path)
        if len(frames) < 2:
            print(f"Skipping {jsonl_path.name}: too few frames ({len(frames)})")
            continue

        stem = jsonl_path.stem
        title = stem.replace("trajectory_", "Task ").replace("_r", " Route ").replace("_ep", " Ep ")

        plot_overview(frames, output_dir / f"{stem}_overview.png", title=title)
        plot_map_view(frames, output_dir / f"{stem}_map.png", title=title)
        print(f"  Plotted: {stem} ({len(frames)} frames)")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
