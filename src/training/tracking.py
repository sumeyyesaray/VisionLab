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


def log_epoch_metrics(epoch: int, train_metrics: dict, val_metrics: dict) -> None:
    mlflow.log_metrics(
        {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        },
        step=epoch,
    )
