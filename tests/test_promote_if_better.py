import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

import promote_if_better


def test_get_production_macro_f1_returns_metrics_when_alias_exists():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="run-1", version="3")
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"val_macro_f1_final": 0.75})
    )

    f1, version = promote_if_better.get_production_macro_f1(client, "mushroom_resnet50")

    assert f1 == 0.75
    assert version == "3"


def test_get_production_macro_f1_returns_none_when_no_production_alias():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("no alias 'production'")

    f1, version = promote_if_better.get_production_macro_f1(client, "mushroom_resnet50")

    assert f1 is None
    assert version is None


def test_main_does_not_promote_when_candidate_is_worse(tmp_path, monkeypatch):
    # RETRAINING_POLICY.md decision 3: this gate is the actual safety net for
    # the flywheel loop, so it's the one behavior worth smoke-testing end to
    # end rather than just at the get_production_macro_f1 level.
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"macro_f1": 0.70}))

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="run-1", version="5")
    fake_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"val_macro_f1_final": 0.85})
    )
    monkeypatch.setattr(promote_if_better, "MlflowClient", lambda: fake_client)
    monkeypatch.setattr(promote_if_better.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_if_better.py",
            "--name",
            "mushroom_resnet50",
            # Never read on this path (main() returns before
            # load_model_from_checkpoint is called), so it doesn't need to
            # point at a real file.
            "--candidate-checkpoint",
            "does/not/need/to/exist.pt",
            "--candidate-report",
            str(report_path),
        ],
    )

    promote_if_better.main()

    fake_client.set_registered_model_alias.assert_not_called()
