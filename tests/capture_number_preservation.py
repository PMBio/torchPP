"""Regenerate the snapshot in ``tests/models/test_number_preservation_snapshot.py``.

Prints a compact (sum, sum-of-squares) fingerprint of every universal-contract
method output for each archetype, on a fixed seed and batch, with the model's
``time_scale`` at its default ``1.0``. Run on CPU and paste the printed
``SNAPSHOT`` dict into the test (only when a change is intended):

    TPP_TEST_DEVICE=cpu PYTHONPATH=.:tests python tests/capture_number_preservation.py
"""

import os

os.environ.setdefault("TPP_TEST_DEVICE", "cpu")  # device-stable capture

import torch  # noqa: E402

from helpers import ARCHETYPES, build_test_data, create_model  # noqa: E402
from torchpp.tpp.base_tpp import IntensityBasedTPP  # noqa: E402


def _fp(x: torch.Tensor):
    """(sum, sum-of-squares) in float64 — a compact bit-identity fingerprint."""
    xd = x.detach().double()
    return (float(xd.sum().item()), float((xd * xd).sum().item()))


def capture() -> dict:
    data = build_test_data(torch.device("cpu"))
    ts = data["batch_multi"]  # B=2 (4 and 5 events)
    B = ts.time_points.shape[0]
    t = torch.full((B,), 3.0, dtype=torch.float64)  # query after the last event
    T = torch.full((B,), 3.0, dtype=torch.float64)  # likelihood window end

    snap: dict = {}
    for name in ARCHETYPES:
        model = create_model(name).to("cpu")
        model.eval()
        out: dict = {}
        with torch.no_grad():
            out["likelihood"] = _fp(model.likelihood(ts, T=T, log=True))
            out["density_time"] = _fp(model.density_time(t, ts, log=True))
            out["density_joint"] = _fp(model.density_joint(t, ts, log=True))
            out["survival_time"] = _fp(model.survival_time(t, ts, log=True))
            out["cdf_time"] = _fp(model.cdf_time(t, ts, log=True))
            out["prob_mark"] = _fp(model.prob_mark(ts, log=True))
            out["prob_mark_given_time"] = _fp(model.prob_mark_given_time(t, ts, log=True))
            out["time_mean"] = _fp(model.time_mean(ts))
            out["time_quantile"] = _fp(model.time_quantile(0.5, ts))
            if isinstance(model, IntensityBasedTPP):
                out["intensity"] = _fp(model.intensity(t, ts))
                out["cumulative_intensity"] = _fp(model.cumulative_intensity(T, ts))
        snap[name] = out
    return snap


def main() -> None:
    snap = capture()
    print("SNAPSHOT = {")
    for name, out in snap.items():
        print(f"    {name!r}: {{")
        for method, fp in out.items():
            print(f"        {method!r}: ({fp[0]!r}, {fp[1]!r}),")
        print("    },")
    print("}")


if __name__ == "__main__":
    main()
