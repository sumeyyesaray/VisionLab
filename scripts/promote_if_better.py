"""The promotion gate: registers a candidate checkpoint (e.g. from a
flywheel retraining run) and only moves the 'production' alias to it if its
macro-F1 actually beats the model currently serving that alias. A
retraining run finishing is not, by itself, a reason to change what's in
production — this is what stops a bad flywheel batch from silently
degrading the live service.

Run at the end of scripts/flywheel_retrain.sbatch, or standalone:

    python scripts/promote_if_better.py \\
        --name mushroom_resnet50 \\
        --candidate-checkpoint outputs/checkpoints/mushroom_resnet50_flywheel_best.pt
"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.checkpoint import load_model_from_checkpoint
from src.training.tracking import MLFLOW_TRACKING_URI, log_mlflow_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Registered model name")
    parser.add_argument("--candidate-checkpoint", required=True, help="Path to a _best.pt checkpoint")
    parser.add_argument(
        "--candidate-report",
        default=None,
        help="Defaults to outputs/reports/<checkpoint-stem-without-_best>_eval_report.json",
    )
    return parser.parse_args()


def get_production_macro_f1(client: MlflowClient, name: str) -> tuple[float | None, str | None]:
    try:
        model_version = client.get_model_version_by_alias(name, "production")
    except MlflowException:
        return None, None
    run = client.get_run(model_version.run_id)
    return run.data.metrics.get("val_macro_f1_final"), model_version.version


def main() -> None:
    args = parse_args()

    checkpoint_stem = Path(args.candidate_checkpoint).stem.removesuffix("_best")
    report_path = Path(
        args.candidate_report or f"outputs/reports/{checkpoint_stem}_eval_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_f1 = report["macro_f1"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    production_f1, production_version = get_production_macro_f1(client, args.name)

    print(f"Candidate macro-F1:  {candidate_f1:.4f}")
    if production_f1 is None:
        print(f"No current 'production' version for {args.name!r}")
    else:
        print(f"Production macro-F1 (v{production_version}): {production_f1:.4f}")

    if production_f1 is not None and candidate_f1 <= production_f1:
        print("Candidate does not beat production — NOT promoting.")
        return

    print("Candidate beats production (or none exists yet) — registering and promoting.")
    model, checkpoint = load_model_from_checkpoint(args.candidate_checkpoint)

    mlflow.set_experiment("visionlab")
    with mlflow.start_run(run_name=f"promote_{args.name}"):
        mlflow.log_metric("val_macro_f1_final", candidate_f1)
        log_mlflow_model(
            model,
            args.candidate_checkpoint,
            registered_model_name=args.name,
            run_metadata=checkpoint.get("run_metadata"),
            label_map=checkpoint.get("label_map"),
            image_size=checkpoint.get("image_size"),
            dataset_type=checkpoint.get("dataset_type"),
        )

    new_version = max(int(mv.version) for mv in client.search_model_versions(f"name='{args.name}'"))
    client.set_registered_model_alias(args.name, "production", str(new_version))
    print(f"Promoted '{args.name}' version {new_version} to production.")


if __name__ == "__main__":
    main()
