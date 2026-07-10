import os
from contextlib import contextmanager

import mlflow

# MLflow's default tracking URI resolves to an absolute sqlite path built
# from the current working directory. On this machine that path contains a
# non-ASCII character (the Windows username), which MLflow's URI encoding
# mishandles and turns into a literal, non-existent directory name — use an
# explicit relative sqlite path instead to sidestep it entirely (MLflow 3.x
# requires a database backend; the plain filesystem store is deprecated).
# Overridable via MLFLOW_TRACKING_URI (e.g. a Google Drive path in Colab,
# where the local disk doesn't survive past the session).
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, params: dict):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        yield run


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
    mlflow.log_metrics(metrics, step=epoch)


def log_run_duration(seconds: float) -> None:
    mlflow.log_metric("run_duration_seconds", seconds)


def log_evaluation_report(report: dict) -> None:
    """Logs macro-F1 as a metric and the worst-classes / most-confused-pairs
    breakdown as a JSON artifact (too detailed to be a scalar metric)."""
    mlflow.log_metric("val_macro_f1_final", report["macro_f1"])
    mlflow.log_dict(report, "evaluation_report.json")
