"""Quick agent diagnostic — runs ~200 steps and reports behavior metrics.

Exit code 0 = agent functional, 1 = agent non-functional.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs/eval_v4/diagnostic")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("ERROR: carla package required")
        sys.exit(2)

    import torch
    from active_inference.config import Config
    from active_inference.agent import DeepAIFAgent
    from active_inference.data.carla_env import CARLADrivingEnv

    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    env = CARLADrivingEnv(host=args.host, port=args.port, town="Town06_Opt")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        spawns = env._world.get_map().get_spawn_points()
        obs = env.reset(spawn_point=spawns[0])
        env.set_goal(spawns[152].location)
        img, state = obs
        agent.reset()

        actions_log = []
        speeds = []
        efe_scores = []
        epistemic_scores = []
        positions = []

        for t in range(args.max_frames):
            img_tensor = torch.tensor(img, dtype=torch.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            plan_result = agent.step_with_info(img_tensor, state_tensor)
            action = plan_result.action

            obs, info = env.step(action.cpu().numpy())
            img, state = obs

            loc = env._vehicle.get_location()
            positions.append((loc.x, loc.y))
            actions_log.append(action.cpu().numpy().tolist())
            speeds.append(float(state[0]))
            efe_scores.append(plan_result.efe_score)
            epistemic_scores.append(plan_result.epistemic_score)
    finally:
        env.close()

    actions_np = np.array(actions_log)
    speeds_np = np.array(speeds)
    positions_np = np.array(positions)

    # Compute distance traveled
    if len(positions_np) > 1:
        diffs = np.diff(positions_np, axis=0)
        total_distance = float(np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)))
    else:
        total_distance = 0.0

    # Speed after warmup (first 20 frames)
    warmup = min(20, len(speeds_np) - 1)
    post_warmup_speed = float(speeds_np[warmup:].mean()) if warmup < len(speeds_np) else 0.0

    functional = post_warmup_speed > 0.5 and total_distance > 5.0

    report = {
        "frames": len(actions_log),
        "total_distance_m": round(total_distance, 2),
        "mean_speed_mps": round(float(speeds_np.mean()), 4),
        "max_speed_mps": round(float(speeds_np.max()), 4),
        "post_warmup_speed": round(post_warmup_speed, 4),
        "mean_steer": round(float(actions_np[:, 0].mean()), 4),
        "mean_abs_steer": round(float(np.abs(actions_np[:, 0]).mean()), 4),
        "mean_accel": round(float(actions_np[:, 1].mean()), 4),
        "mean_abs_accel": round(float(np.abs(actions_np[:, 1]).mean()), 4),
        "std_steer": round(float(actions_np[:, 0].std()), 4),
        "std_accel": round(float(actions_np[:, 1].std()), 4),
        "mean_efe": round(float(np.mean(efe_scores)), 6),
        "std_efe": round(float(np.std(efe_scores)), 6),
        "mean_epistemic": round(float(np.mean(epistemic_scores)), 6),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "functional": functional,
    }

    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if functional:
        print("\n*** DIAGNOSTIC: Agent FUNCTIONAL ***")
        sys.exit(0)
    else:
        print("\n*** DIAGNOSTIC: Agent NOT FUNCTIONAL ***")
        print(f"  post_warmup_speed={post_warmup_speed:.3f} m/s (need >0.5)")
        print(f"  total_distance={total_distance:.1f} m (need >5)")
        sys.exit(1)


if __name__ == "__main__":
    main()
