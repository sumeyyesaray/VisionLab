"""Checks whether enough verified flywheel predictions (see POST /feedback
in src/deployment/api.py) have accumulated to justify a retraining run, and
submits scripts/flywheel_retrain.sbatch if so.

Not started automatically by anything in this repo — run it manually, or
schedule it yourself (e.g. a login-node crontab entry) once you're
comfortable with what it does:

    0 3 * * * cd /arf/scratch/hasozkan/sum/VisionLab && \\
        python scripts/check_flywheel_threshold.py --dataset-type mushroom
"""

import argparse
import subprocess

from src.flywheel.store import count_verified_unused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", required=True, choices=["mushroom", "flower"])
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Minimum verified, not-yet-used predictions before retraining is worth it",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the count but don't submit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = count_verified_unused(args.dataset_type)
    print(f"{args.dataset_type}: {n} verified, unused flywheel predictions")

    if n < args.threshold:
        print(f"Below threshold ({args.threshold}) — nothing to do")
        return

    if args.dry_run:
        print(f"Threshold reached ({n} >= {args.threshold}) — dry run, not submitting")
        return

    subprocess.run(["sbatch", "scripts/flywheel_retrain.sbatch", args.dataset_type], check=True)
    print("Submitted scripts/flywheel_retrain.sbatch")


if __name__ == "__main__":
    main()
