"""Checks whether it's time to retrain, per RETRAINING_POLICY.md decision 2:

    verified_count >= 500  OR  days_since_last_check >= 30

Either condition alone is enough to submit scripts/flywheel_retrain.sbatch —
the count threshold catches "a lot of new data piled up fast," the day
threshold catches "barely anything came in, but it's been a month, so check
anyway" (a monthly sanity retrain, harmless even with little new data since
scripts/promote_if_better.py won't promote a worse or no-op result).

Not started automatically by anything in this repo — run it manually, or
schedule it yourself (e.g. a login-node crontab entry) once you're
comfortable with what it does:

    0 3 * * * cd /arf/scratch/hasozkan/sum/VisionLab && \\
        python scripts/check_flywheel_threshold.py --dataset-type mushroom
"""

import argparse
import subprocess

from src.flywheel.store import count_verified_unused, days_since_last_trigger, record_retrain_trigger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", required=True, choices=["mushroom", "flower"])
    parser.add_argument(
        "--count-threshold",
        type=int,
        default=500,
        help="Retrain if at least this many verified, not-yet-used predictions exist",
    )
    parser.add_argument(
        "--days-threshold",
        type=float,
        default=30,
        help="Retrain if it's been at least this many days since the last trigger, "
        "regardless of count",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the decision but don't submit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = count_verified_unused(args.dataset_type)
    days = days_since_last_trigger(args.dataset_type)

    count_reason = n >= args.count_threshold
    days_reason = days is None or days >= args.days_threshold

    days_display = "never triggered before" if days is None else f"{days:.1f} days ago"
    print(f"{args.dataset_type}: {n} verified/unused (threshold {args.count_threshold}); "
          f"last trigger {days_display} (threshold {args.days_threshold} days)")

    if not (count_reason or days_reason):
        print("Neither condition met — nothing to do")
        return

    reason = "count" if count_reason else "time"
    print(f"Retraining condition met ({reason}-based)")

    if args.dry_run:
        print("Dry run — not submitting")
        return

    subprocess.run(["sbatch", "scripts/flywheel_retrain.sbatch", args.dataset_type], check=True)
    record_retrain_trigger(args.dataset_type)
    print("Submitted scripts/flywheel_retrain.sbatch")


if __name__ == "__main__":
    main()
