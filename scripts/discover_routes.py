"""Discover evaluation routes for Task B (obstacle avoidance).

Scans spawn-point pairs to find intersection-free, multi-lane segments
suitable for obstacle placement. Results are printed for copy-paste
into src/active_inference/evaluation/routes.py.

Usage:
    uv run python scripts/discover_routes.py --town Town06_Opt --task obstacle_avoidance
    uv run python scripts/discover_routes.py --town Town06_Opt --task curves
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Discover evaluation routes in a CARLA town"
    )
    parser.add_argument("--town", required=True, help="CARLA town name (e.g., Town06_Opt)")
    parser.add_argument(
        "--task",
        choices=["obstacle_avoidance", "curves"],
        default="obstacle_avoidance",
        help="Route type to discover",
    )
    parser.add_argument("--min_distance", type=float, default=300.0)
    parser.add_argument("--max_distance", type=float, default=800.0)
    parser.add_argument("--max_results", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("ERROR: carla package required")
        sys.exit(1)

    from active_inference.evaluation.routes import (
        find_routes_with_curves,
        find_straight_routes,
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"Discovering {args.task} routes in {args.town}...")
    print(f"  Distance range: {args.min_distance}-{args.max_distance}m")
    print()

    if args.task == "obstacle_avoidance":
        results = find_straight_routes(
            client,
            args.town,
            max_turns=5,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_results=args.max_results,
        )
    else:
        results = find_routes_with_curves(
            client,
            args.town,
            min_turns=2,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            max_results=args.max_results,
        )

    if not results:
        print("No routes found matching criteria.")
        sys.exit(0)

    print(f"\nFound {len(results)} routes. Copy-paste into routes.py:\n")
    route_key = f"{args.town}_TaskB" if args.task == "obstacle_avoidance" else args.town
    print(f'    "{route_key}": [')
    for r in results:
        desc = f"{'Highway obstacle avoidance' if args.task == 'obstacle_avoidance' else 'Curves'} {r['length']}m {r['turns']}turns"
        print(f'        {{"start": {r["start"]}, "goal": {r["goal"]}, "desc": "{desc}"}},')
    print("    ],")


if __name__ == "__main__":
    main()
