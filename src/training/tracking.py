from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.pytorch
import wandb


@contextmanager
def wandb_run(project_name: str, run_name: str, params: dict):
    run = wandb.init(project=project_name, name=run_name, config=params)
    try:
        yield run
    finally:
        wandb.finish()


def log_epoch_metrics(
    epoch: int, train_metrics: dict, val_metrics: dict, extra_metrics: dict | None = None
) -> None:
    metrics = {
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    wandb.log(metrics, step=epoch)


def log_run_duration(seconds: float) -> None:
    wandb.run.summary["run_duration_seconds"] = seconds


def log_evaluation_report(report: dict) -> None:
    """Logs macro-F1 as a summary metric and the worst-classes /
    most-confused-pairs breakdown as W&B tables — sortable/filterable in
    the dashboard, unlike a flat JSON artifact."""
    wandb.run.summary["val_macro_f1_final"] = report["macro_f1"]

    worst_classes_table = wandb.Table(columns=["label", "precision", "recall", "f1", "support"])
    for row in report["worst_classes"]:
        worst_classes_table.add_data(
            row["label"], row["precision"], row["recall"], row["f1"], row["support"]
        )

    confused_pairs_table = wandb.Table(columns=["true_label", "predicted_label", "count"])
    for row in report["top_confused_pairs"]:
        confused_pairs_table.add_data(row["true_label"], row["predicted_label"], row["count"])

    wandb.log({"worst_classes": worst_classes_table, "top_confused_pairs": confused_pairs_table})


# --- MLflow -----------------------------------------------------------------
# Runs alongside W&B, not instead of it: W&B stays the tool for comparing
# experiments (its dashboards/tables are built for that), MLflow answers
# "which model is in production" via its Model Registry. No dedicated
# tracking server on TRUBA, so this is a local backend — but MLflow 3.x's
# plain `file:./mlruns` store is in maintenance mode and refuses new runs
# (raises MlflowException), so this uses the SQLite backend it now expects
# instead. Artifacts (checkpoints, registered models) still land in
# `mlartifacts/` next to it — both already gitignored. Inspect with
# `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
#
# Caveat: SQLite serializes writes, so two training jobs finishing an epoch
# at the same moment could hit "database is locked" — hasn't been an issue
# at this project's concurrency (2 simultaneous jobs), but a real tracking
# server would be needed before running many jobs in parallel.

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, params: dict, run_metadata: dict | None = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        if run_metadata:
            mlflow.set_tags({k: str(v) for k, v in run_metadata.items() if v is not None})
        yield run


def log_epoch_metrics_mlflow(
    epoch: int, train_metrics: dict, val_metrics: dict, extra_metrics: dict | None = None
) -> None:
    metrics = {
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }
    if extra_metrics:
        metrics.update({k: v for k, v in extra_metrics.items() if isinstance(v, (int, float))})
    mlflow.log_metrics(metrics, step=epoch)


def log_mlflow_model(
    model,
    checkpoint_path: str | Path,
    registered_model_name: str | None = None,
    run_metadata: dict | None = None,
) -> None:
    """Logs the resumable checkpoint file (optimizer/scaler state included)
    as a plain artifact, and separately registers the model itself in
    MLflow's Model Registry via `mlflow.pytorch.log_model` — that's what
    makes it loadable with `mlflow.pytorch.load_model` and visible in the
    registry's "which version is production" view, which a raw artifact
    file alone wouldn't give you.
    """
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")
    tags = {k: str(v) for k, v in (run_metadata or {}).items() if v is not None}
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        tags=tags or None,
        # Default 'pt2' format traces the model via torch.export and requires
        # an input_example; our custom architectures (e.g. the hierarchical
        # genus+species model returning a tuple) aren't guaranteed export-
        # traceable, so stick with the classic pickle-based save.
        serialization_format="pickle",
    )


def log_mlflow_evaluation_report(report: dict) -> None:
    mlflow.log_metric("val_macro_f1_final", report["macro_f1"])
    mlflow.log_dict(report, "evaluation_report.json")
