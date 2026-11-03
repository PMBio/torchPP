"""Conversion of EasyTPP json files (Hugging Face layout) in the benchmark download script."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data_prep" / "download_benchmark_datasets.py"


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_benchmark_datasets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(json_dir, rows):
    json_dir.mkdir()
    for split in ("train", "dev", "test"):
        (json_dir / f"{split}.json").write_text(json.dumps(rows))


def test_json_conversion_keeps_float64_times(dl, tmp_path):
    times = [0.0, 1e-4, 2.0000001, 3.5]
    rows = [{"dim_process": 3, "seq_len": 4, "time_since_start": times, "type_event": [0, 2, 1, 0]}]
    _write(tmp_path / "json", rows)
    out = tmp_path / "out"
    out.mkdir()
    dl.load_from_json(tmp_path / "json", 3, out, dtype=torch.float64)
    for split in ("train", "dev", "test"):
        (seq,) = torch.load(out / f"{split}.pt", weights_only=False)
        assert seq.time_points.dtype == torch.float64
        assert seq.time_points.tolist() == times
        assert seq.event_types.tolist() == [0, 2, 1, 0]


def test_json_conversion_checks_num_types(dl, tmp_path):
    rows = [{"dim_process": 5, "seq_len": 1, "time_since_start": [0.0], "type_event": [0]}]
    _write(tmp_path / "json", rows)
    with pytest.raises(AssertionError):
        dl.load_from_json(tmp_path / "json", 3, tmp_path)


def test_taobao_is_registered_as_float64_json(dl):
    assert dl.DATASETS["taobao"] == ("easytpp/taobao", 17)
    assert dl.JSON_DATASETS["taobao"] == torch.float64
