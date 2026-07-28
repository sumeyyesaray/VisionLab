"""Backfills an existing checkpoint (trained before MLflow integration
existed, or trained with --no-mlflow) into the MLflow Model Registry, so it
can be served via src/inference/registry.py.

Newer checkpoints (saved after src/models/checkpoint.py's `label_map`/
`image_size` fields were added) carry everything needed automatically. Older
ones need `--label-map`/`--image-size` passed explicitly since those fields
don't exist in the checkpoint file yet.

Examples:

    python scripts/register_model.py \\
        --checkpoint outputs/checkpoints/mushroom_resnet50_twostage_stage1_best.pt \\
        --name mushroom_resnet50 --image-size 384 --production

    python scripts/register_model.py \\
        --checkpoint outputs/checkpoints/flower_resnet50_best.pt \\
        --name flower_resnet50 --image-size 224 --production
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.data.label_map import load_label_map
from src.models.checkpoint import load_checkpoint, load_model_from_checkpoint
from src.training.tracking import MLFLOW_TRACKING_URI, log_mlflow_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a _best.pt checkpoint")
    parser.add_argument("--name", required=True, help="Registered model name (e.g. mushroom_resnet50)")
    parser.add_argument(
        "--label-map",
        default=None,
        help="Path to a label_map JSON, if the checkpoint predates embedding it directly",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Inference image size, if the checkpoint predates embedding it directly",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Also point the 'production' alias at the newly registered version",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model_from_checkpoint(args.checkpoint)

    label_map = checkpoint.get("label_map")
    if label_map is None:
        if args.label_map:
            label_map = load_label_map(args.label_map)
        else:
            label_map = load_label_map(checkpoint["label_map_path"])

    image_size = checkpoint.get("image_size") or args.image_size
    if image_size is None:
        raise ValueError(
            "Checkpoint has no embedded image_size — pass --image-size explicitly."
        )

    run_metadata = checkpoint.get("run_metadata", {})

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("visionlab")
    with mlflow.start_run(run_name=f"register_{args.name}"):
        mlflow.log_params(
            {
                "architecture": checkpoint["architecture"],
                "dataset_type": checkpoint["dataset_type"],
                "source_checkpoint": args.checkpoint,
                "backfilled": True,
            }
        )
        log_mlflow_model(
            model,
            args.checkpoint,
            registered_model_name=args.name,
            run_metadata=run_metadata,
            label_map=label_map,
            image_size=image_size,
            dataset_type=checkpoint["dataset_type"],
        )

    if args.production:
        client = MlflowClient()
        latest_version = max(
            int(mv.version) for mv in client.search_model_versions(f"name='{args.name}'")
        )
        client.set_registered_model_alias(args.name, "production", str(latest_version))
        print(f"'{args.name}' version {latest_version} is now aliased 'production'")

    print(f"Registered '{args.name}' from {args.checkpoint}")


if __name__ == "__main__":
    main()
