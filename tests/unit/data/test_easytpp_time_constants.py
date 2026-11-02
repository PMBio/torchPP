"""The EasyTPP time constants: defaults unchanged, root override for the HuggingFace float64 Taobao root."""
import torch

from torchpp.data.easytpp_data import (
    EASYTPP_MEAN_IET,
    EASYTPP_TIME_SCALE,
    EasyTPPDataset,
    easytpp_time_constants,
)
from torchpp.utils.event_data import MVEventData


def test_taobao_default_constants_unchanged():
    # the defaults (Google Drive Taobao release) are unchanged
    assert EASYTPP_MEAN_IET["taobao"] == 0.8081
    assert EASYTPP_TIME_SCALE["taobao"] == 46.7464
    assert easytpp_time_constants("taobao") == (0.8081, 46.7464)
    assert easytpp_time_constants("taobao", "easytpp") == (0.8081, 46.7464)
    assert easytpp_time_constants("taobao", "easytpp_f64") == (0.8081, 46.7464)


def test_taobao_hf_root_override():
    assert easytpp_time_constants("taobao", "easytpp_hf_f64") == (0.0463, 2.6197)
    # other datasets are unaffected by the root
    assert easytpp_time_constants("taxi", "easytpp_hf_f64") == (0.2246, 8.3182)


def test_loader_applies_root_override(tmp_path):
    seqs = [MVEventData(time_points=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
                        event_types=torch.tensor([0, 1, 2]))]
    for root, (iet, tsf) in [("easytpp_hf_f64", (0.0463, 2.6197)), ("easytpp", (0.8081, 46.7464))]:
        d = tmp_path / root / "taobao"
        d.mkdir(parents=True)
        torch.save(seqs, d / "train.pt")
        ds = EasyTPPDataset.load({"dataset_name": "taobao", "easytpp_root": root}, tmp_path, "train")
        assert ds.time_scale_factor == tsf
        tp = ds[0]["time_points"]
        assert tp.dtype == torch.float64
        assert torch.allclose(tp, (torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64) + iet) / tsf)
