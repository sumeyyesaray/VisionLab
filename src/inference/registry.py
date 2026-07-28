"""Looks up servable models from the MLflow Model Registry — the boundary
between "a model exists somewhere in outputs/checkpoints" and "a model the
serving layer will actually load." Only registered-model versions carrying
the 'production' alias (see scripts/register_model.py's --production flag,
or scripts/train_baseline.py's automatic registration) are servable; being
in the registry at all isn't enough, since every training run registers a
candidate version but promotion to 'production' is a separate, deliberate
step.
"""

import json
from dataclasses import dataclass

import mlflow.pytorch
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.training.tracking import MLFLOW_TRACKING_URI

PRODUCTION_ALIAS = "production"


@dataclass
class ServableModel:
    name: str
    version: str
    model: torch.nn.Module
    idx_to_label: dict[int, str]
    image_size: int
    dataset_type: str


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def list_production_model_names() -> list[str]:
    """Every registered model name that currently has a 'production' alias
    pointing at some version — i.e. what /models should advertise."""
    client = _client()
    names = []
    for registered_model in client.search_registered_models():
        try:
            client.get_model_version_by_alias(registered_model.name, PRODUCTION_ALIAS)
        except MlflowException:
            continue
        names.append(registered_model.name)
    return sorted(names)


def load_production_model(name: str) -> ServableModel:
    """Loads the model currently aliased 'production' for `name`, along with
    the class mapping and preprocessing size that were tagged onto that
    exact model version at registration time (see
    src/training/tracking.log_mlflow_model) — nothing here is hardcoded per
    domain, so a new dataset only needs a new registered model, not a code
    change.
    """
    client = _client()
    try:
        model_version = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        raise KeyError(f"No '{PRODUCTION_ALIAS}' alias set for model {name!r}") from exc

    model = mlflow.pytorch.load_model(f"models:/{name}@{PRODUCTION_ALIAS}")
    model.eval()

    label_map = json.loads(model_version.tags["label_map"])  # label (str) -> index (int)
    idx_to_label = {idx: label for label, idx in label_map.items()}
    image_size = int(model_version.tags["image_size"])
    # Fallback covers versions registered before the dataset_type tag existed —
    # every registered_model_name follows the f"{dataset_type}_{model}" convention.
    dataset_type = model_version.tags.get("dataset_type") or name.split("_")[0]

    return ServableModel(
        name=name,
        version=model_version.version,
        model=model,
        idx_to_label=idx_to_label,
        image_size=image_size,
        dataset_type=dataset_type,
    )


def load_all_production_models() -> dict[str, ServableModel]:
    return {name: load_production_model(name) for name in list_production_model_names()}
