import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/debug.yaml")
    parser.add_argument("--output_dir", default="outputs/pipeline")
    parser.add_argument("--skip_collect", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_collect:
        run(
            [
                python,
                "scripts/collect_data.py",
                "--num_samples",
                "100",
                "--output",
                f"{args.output_dir}/data.h5",
            ]
        )

    run(
        [
            python,
            "scripts/train.py",
            "--config",
            args.config,
            "--epochs",
            "2",
            "--output_dir",
            f"{args.output_dir}/train",
            "--data",
            f"{args.output_dir}/data.h5" if not args.skip_collect else "",
        ]
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
