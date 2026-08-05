import json

from src.flywheel import export as export_module


def _record(id_, true_label, image_path):
    # export_flywheel_snapshot only ever does dict-style subscript access
    # (record["id"], record["true_label"], record["image_path"]), which
    # sqlite3.Row supports too — so a plain dict is a valid stand-in here.
    return {"id": id_, "true_label": true_label, "image_path": str(image_path)}


def test_export_happy_path(tmp_path, monkeypatch):
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(json.dumps({"cat": 0, "dog": 1}))

    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"fake-image-bytes-1")
    img2 = tmp_path / "img2.jpg"
    img2.write_bytes(b"fake-image-bytes-2")

    records = [_record(1, "cat", img1), _record(2, "dog", img2)]
    monkeypatch.setattr(export_module, "get_verified_unused", lambda dataset_type: records)
    marked_ids = []
    monkeypatch.setattr(export_module, "mark_used_in_training", marked_ids.extend)

    output_root = tmp_path / "snapshot"
    csv_path = export_module.export_flywheel_snapshot("mushroom", str(label_map_path), output_root)

    assert csv_path == output_root / "train.csv"
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 rows
    assert (output_root / "images" / "1.jpg").exists()
    assert (output_root / "images" / "2.jpg").exists()
    assert sorted(marked_ids) == [1, 2]


def test_export_skips_records_with_unknown_or_missing_data(tmp_path, monkeypatch):
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(json.dumps({"cat": 0}))

    img1 = tmp_path / "img1.jpg"
    img1.write_bytes(b"fake-image-bytes")

    records = [
        _record(1, "cat", img1),  # valid
        _record(2, "unknown_species", img1),  # true_label not in label_map
        _record(3, "cat", tmp_path / "missing.jpg"),  # source file doesn't exist
    ]
    monkeypatch.setattr(export_module, "get_verified_unused", lambda dataset_type: records)
    marked_ids = []
    monkeypatch.setattr(export_module, "mark_used_in_training", marked_ids.extend)

    output_root = tmp_path / "snapshot"
    csv_path = export_module.export_flywheel_snapshot("mushroom", str(label_map_path), output_root)

    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 2  # header + only the one valid row
    assert marked_ids == [1]


def test_export_returns_none_when_no_verified_records(tmp_path, monkeypatch):
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(json.dumps({"cat": 0}))
    monkeypatch.setattr(export_module, "get_verified_unused", lambda dataset_type: [])

    result = export_module.export_flywheel_snapshot("mushroom", str(label_map_path), tmp_path / "snapshot")

    assert result is None
