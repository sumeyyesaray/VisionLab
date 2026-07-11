from contextlib import contextmanager

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
