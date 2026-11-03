"""Input validation, training-loop behaviour, config strictness and factory defaults."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from torchpp import BatchedMVEventData, MVEventData, NeuralSplineProcess, TokenRoles, create_model
from torchpp.modules.intensity import ExpKernelModule, KernelIntensityModule
from torchpp.training.config import BatchingConfig
from torchpp.training.config_utils import ConfigError, load_config
from torchpp.training.config import FitConfig
from torchpp.training.trainer import _aggregate_val_ll
from torchpp.training.training_helpers import make_collate_fns
from torchpp.utils.batching import BucketBatchSampler, print_length_analysis
from torchpp.utils.evaluation import evaluate_tpp


def _seq(t=(0.4, 1.1, 1.3, 2.7), m=(0, 1, 1, 0), **kw):
    return MVEventData(torch.tensor(t), torch.tensor(m), **kw)


# ----------------------------------------------------------------------------- inputs


def test_decreasing_times_raise_and_sort_option():
    with pytest.raises(ValueError, match="non-decreasing"):
        _seq((1.4, 0.1), (0, 1))
    s = _seq((1.4, 0.1), (0, 1), sort=True)
    assert s.time_points.tolist() == pytest.approx([0.1, 1.4]) and s.event_types.tolist() == [1, 0]


@pytest.mark.parametrize(
    "times, marks, err",
    [
        (torch.tensor([0.4, 3.1]), torch.tensor([0]), ValueError),
        (torch.tensor([0.4, 1.0]), torch.tensor([0.0, 1.0]), TypeError),
        (torch.tensor([[0.4, 1.0]]), torch.tensor([[0, 1]]), ValueError),
        (torch.tensor([0.4, 1.0]), torch.tensor([0, -3]), ValueError),
    ],
)
def test_malformed_sequences_raise(times, marks, err):
    with pytest.raises(err):
        MVEventData(times, marks)


def test_lists_are_converted():
    s = MVEventData([0.4, 1.0], [0, 1])
    assert s.time_points.dtype.is_floating_point and s.event_types.dtype == torch.long


def test_T_obs_before_last_event_warns():
    with pytest.warns(UserWarning, match="before the last event"):
        s = _seq(T_obs=2.0)
    assert s.T_obs == pytest.approx(2.7)


def test_event_type_out_of_range_raises():
    model = create_model("const_exp_hawkes", D=2, seed=0)
    with pytest.raises(ValueError, match="out of range"):
        model.likelihood(_seq(m=(0, 1, 5, 0)))


def test_single_sequence_and_scalar_times_are_accepted():
    model = create_model("const_exp_hawkes", D=2, seed=0)
    s = _seq(T_obs=3.0)
    b = s.to_batch()
    assert torch.equal(model.likelihood(s), model.likelihood(b))
    assert torch.equal(model.intensity(1.0, s), model.intensity(torch.tensor([1.0]), b))
    with pytest.raises(ValueError, match="t >= 0"):
        model.intensity(-1.0, s)
    with pytest.raises(TypeError, match="BatchedMVEventData"):
        model.likelihood("not a sequence")


# ----------------------------------------------------------------------------- training loop


def test_validation_runs_in_eval_mode_and_restores_mode():
    model = create_model("neural_spline", D=2, seed=0)  # dropout 0.1 by default
    model.train()
    batch = BatchedMVEventData(mv_events=[_seq(), _seq((0.2, 0.9), (1, 0))])
    v1, _ = _aggregate_val_ll(model, [batch], "cpu")
    v2, _ = _aggregate_val_ll(model, [batch], "cpu")
    assert v1 == v2 and model.training
    model.eval()
    with torch.no_grad():
        expected = float(model.likelihood(batch).sum() / model.num_scored_events(batch).sum())
    assert v1 == pytest.approx(expected, rel=1e-6)


def test_bucket_sampler_reshuffles_each_pass():
    data = [_seq(tuple(np.cumsum(np.full(n, 0.1))), tuple([0] * n)) for n in range(1, 41)]
    order = lambda s: [tuple(b) for b in s]  # noqa: E731
    s1 = BucketBatchSampler(data, batch_size=4, shuffle=True, seed=3)
    first, second = order(s1), order(s1)
    assert first != second
    s2 = BucketBatchSampler(data, batch_size=4, shuffle=True, seed=3)
    assert order(s2) == first


def test_random_truncation_new_crop_per_call_reproducible():
    seq = _seq(tuple(np.cumsum(np.full(30, 0.1))), tuple([0] * 30))
    cfg = BatchingConfig(max_sequence_length=5, truncation_strategy="random")
    train_a, val = make_collate_fns(cfg, seed=0)
    crops = [tuple(train_a([seq]).time_points[0].tolist()) for _ in range(3)]
    assert len(set(crops)) > 1
    train_b, _ = make_collate_fns(cfg, seed=0)
    assert tuple(train_b([seq]).time_points[0].tolist()) == crops[0]
    assert torch.equal(val([seq]).time_points, val([seq]).time_points)


# ----------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "argv, match",
    [
        (["training.num_steps", "50"], "missing '='"),
        (["--confg", "x"], "Unknown option"),
        (["training.num_steps=abc"], "must be int"),
        (["training.num_step=50"], "Did you mean 'training.num_steps'"),
        (["not_a_section=1"], "Unknown key 'not_a_section'"),
        (["dataset.datset_name=taxi"], "Did you mean 'dataset.dataset_name'"),
    ],
)
def test_config_errors(argv, match):
    base = ["-c", "default_fit", "-c", "model/const_exp_hawkes", "-c", "data/taxi"]
    with pytest.raises(ValueError, match=match):
        load_config(FitConfig, argv=base + argv)


def test_help_exits(capsys):
    with pytest.raises(SystemExit):
        load_config(FitConfig, argv=["--help"])
    assert "Presets in" in capsys.readouterr().out


# ----------------------------------------------------------------------------- factory + seeding


def test_create_model_defaults(capsys):
    model = create_model("hawkes", D=2)
    assert not model.terminating and model.run_name == "const_exp_hawkes"
    assert capsys.readouterr().out == ""


def test_neural_seed_controls_init_and_leaves_global_rng():
    torch.manual_seed(0)
    a = NeuralSplineProcess(D=3, seed=1)
    torch.manual_seed(5)
    b = NeuralSplineProcess(D=3, seed=1)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    torch.manual_seed(0)
    ref = torch.rand(1)
    torch.manual_seed(0)
    create_model("neural_spline", D=3, seed=3)
    assert torch.equal(torch.rand(1), ref)


# ----------------------------------------------------------------------------- smaller fixes


def test_clamping_fn_case_insensitive():
    from torchpp.tpp.hawkes import InhibitiveConstBaselineExpKernelHawkesProcess

    upper = InhibitiveConstBaselineExpKernelHawkesProcess(D=2, seed=0, clamping_fn="ReLU")
    ref = InhibitiveConstBaselineExpKernelHawkesProcess(D=2, seed=0, clamping_fn="relu")
    assert all(torch.equal(p, q) for p, q in zip(upper.state_dict().values(), ref.state_dict().values()))


def test_default_kernel_event_intensities_handle_padding():
    kernel = ExpKernelModule(D=2, seed=0)
    batch = BatchedMVEventData(mv_events=[_seq(), _seq((0.2,), (1,))])
    out = KernelIntensityModule.event_intensities(kernel, batch)
    assert out.shape == batch.time_points.shape and torch.isfinite(out).all()
    assert torch.allclose(out[batch.event_mask], kernel.event_intensities(batch)[batch.event_mask])


def test_checkpoint_without_time_scale_raises(tmp_path):
    model = create_model("poisson", D=2, seed=0)
    state = {k: v for k, v in model.state_dict().items() if k != "time_scale"}
    path = tmp_path / "old.pth"
    torch.save(state, path)
    with pytest.raises(ValueError, match="no time_scale"):
        create_model("poisson", D=2, load_path=path)


def test_print_length_analysis_empty(capsys):
    print_length_analysis([])
    assert "No sequences" in capsys.readouterr().out


def test_evaluate_tpp_contract():
    model = create_model("const_exp_hawkes", D=2, seed=0)
    model.train()
    metrics = evaluate_tpp(model, [_seq(), _seq((0.2, 0.9, 1.4), (1, 0, 1))], compute_marginals=False)
    assert metrics["num_predictions"] == 5 and model.training
    single = evaluate_tpp(model, [_seq((0.4,), (0,))], compute_marginals=False)
    assert single["num_predictions"] == 0 and single["num_scored_events"] == 0
    with pytest.raises(ValueError, match="no sequences"):
        evaluate_tpp(model, [], compute_marginals=False)


def test_density_eval_excludes_context_types():
    seqs = [_seq((0.4, 1.1, 1.3, 2.7), (0, 1, 1, 0)), _seq((0.2, 0.9, 1.4), (1, 0, 1))]
    model = create_model("intensity_free", D=3, seed=0, token_roles=TokenRoles(context_types=[2]))
    implicit = evaluate_tpp(model, seqs, compute_marginals=False)
    explicit = evaluate_tpp(model, seqs, roles=model.token_roles.replace(ignored_types=(2,)), compute_marginals=False)
    assert implicit["cond_type_ll"] == pytest.approx(explicit["cond_type_ll"])
    assert implicit["joint_ll"] == pytest.approx(explicit["joint_ll"])


def test_sample_carries_no_grad():
    model = create_model("const_exp_hawkes", D=2, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out, _, _ = model.sample(_seq(T_obs=3.0), num_steps=2, rng=0)
    assert not out.time_points.requires_grad and float(out.time_points[-2]) > 3.0


# ----------------------------------------------------------------------------- release review fixes


@pytest.mark.parametrize(
    "times, marks",
    [
        (torch.tensor([0.4, 1.0, 1.3]), torch.tensor([0, 1, -1])),  # padding outside a batch
        (torch.tensor([0.4, float("nan")]), torch.tensor([0, 1])),
        (torch.tensor([0.4, float("inf")]), torch.tensor([0, 1])),
    ],
)
def test_single_sequence_rejects_padding_and_non_finite_times(times, marks):
    with pytest.raises(ValueError):
        MVEventData(times, marks)


def test_batch_rows_still_accept_trailing_padding():
    b = BatchedMVEventData([torch.tensor([0.4, 1.0, 1.3])], [torch.tensor([0, 1, -1])])
    assert b.real_counts.tolist() == [2]


def test_bucket_sampler_with_equal_lengths():
    seqs = [MVEventData(torch.arange(1.0, 9.0), torch.zeros(8, dtype=torch.long))] * 20
    batches = list(BucketBatchSampler(seqs, batch_size=4, seed=0))
    assert sorted(i for b in batches for i in b) == list(range(20))
    dyn = BucketBatchSampler(seqs, batch_size=4, seed=0, dynamic_batch_size=True, target_total_events=16)
    assert sum(len(b) for b in dyn) == 20


def _toy_dataset_cls(name, cfg_keys=frozenset({"path"})):
    from torchpp.data import TPPDataset
    from torchpp.training.dataset_registry import DATASET_REGISTRY, register_dataset

    DATASET_REGISTRY.pop(name, None)

    @register_dataset(name)
    class Toy(TPPDataset):
        def __init__(self, n):
            self.seqs = [_seq() for _ in range(n)]

        @property
        def num_event_types(self):
            return 2

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, i):
            return self.seqs[i]

        @classmethod
        def load(cls, cfg, root, split):
            assert "limit_test_size" not in cfg  # applied by the registry
            return cls(30)

    Toy.cfg_keys = cfg_keys
    return Toy


def test_limit_keys_apply_to_every_dataset(tmp_path):
    from torchpp.training.dataset_registry import DATASET_REGISTRY, load_split

    _toy_dataset_cls("toy_limit")
    try:
        ds = load_split("toy_limit", {"path": "x", "limit_test_size": 7}, tmp_path, "test")
        assert len(ds) == 7 and ds.num_event_types == 2 and ds.terminating is False
        assert len(list(ds)) == 7 and ds[-1] is ds._inner[6]
        assert len(load_split("toy_limit", {"path": "x", "limit_test_size": 7}, tmp_path, "train")) == 30
    finally:
        DATASET_REGISTRY.pop("toy_limit", None)


def test_dataset_module_is_imported_before_validation(tmp_path, monkeypatch):
    from torchpp.training.dataset_registry import DATASET_REGISTRY

    (tmp_path / "my_toy_data.py").write_text(
        "import torch\n"
        "from torchpp import MVEventData\n"
        "from torchpp.data import TPPDataset\n"
        "from torchpp.training.dataset_registry import register_dataset\n"
        "@register_dataset('my_toy_data')\n"
        "class D(TPPDataset):\n"
        "    cfg_keys = frozenset({'path'})\n"
        "    num_event_types = 2\n"
        "    def __len__(self): return 0\n"
        "    def __getitem__(self, i): raise IndexError\n"
        "    @classmethod\n"
        "    def load(cls, cfg, root, split): return cls()\n"
    )
    monkeypatch.chdir(tmp_path)
    try:
        configs = Path(__file__).resolve().parents[2] / "configs"
        base = ["-c", "default_fit", "model_type=poisson"]
        cfg = load_config(
            FitConfig,
            argv=base + ["dataset_module=my_toy_data", "dataset_type=my_toy_data", "dataset.path=x"],
            configs_dir=configs,
        )
        assert cfg.dataset_type == "my_toy_data"
        with pytest.raises(ConfigError, match="could not be imported"):
            load_config(FitConfig, argv=base + ["dataset_module=no_such_module"], configs_dir=configs)
    finally:
        DATASET_REGISTRY.pop("my_toy_data", None)


def test_create_model_accepts_arch_dict_and_rejects_typos():
    assert create_model("spline_poisson", D=2, arch={"baseline": {"num_knots": 3}}, seed=0).num_knots == 3
    with pytest.raises(ValueError, match="Did you mean 'arch.baseline.num_knots'"):
        create_model("spline_poisson", D=2, arch={"baseline": {"num_knot": 3}})
    with pytest.raises(ValueError, match="has no architecture field 'arch.encoder'"):
        create_model("spline_poisson", D=2, arch={"encoder": {"n_layers": 2}})


def test_save_model_infers_type_and_checks_arguments(tmp_path):
    from torchpp import load_model, save_model

    model = create_model("hawkes", D=2, seed=0)
    loaded = load_model(save_model(tmp_path / "m.pth", model))
    assert type(loaded) is type(model) and loaded.model_type == "const_exp_hawkes"
    with pytest.raises(ValueError, match="does not match"):
        save_model(tmp_path / "x.pth", model, model_type="poisson")
    with pytest.raises(ValueError, match="does not match"):
        save_model(tmp_path / "x.pth", model, D=3)


def test_poisson_constructors_take_D_first():
    from torchpp import LogLinearPoissonProcess, PoissonProcess
    from torchpp.tpp.baseline import PoissonProcessParams

    assert PoissonProcess(3).D == 3 and LogLinearPoissonProcess(2, seed=0).D == 2
    with pytest.raises(TypeError, match="params="):
        PoissonProcess(PoissonProcessParams(mu=torch.tensor([1.0])))


@pytest.mark.parametrize("model_type", ["const_exp_hawkes", "intensity_free"])
def test_sample_accepts_a_one_row_batch(model_type):
    model = create_model(model_type, D=2, seed=0).eval()  # dropout off: same rng -> same draw
    seq = _seq()
    out, times, _ = model.sample(BatchedMVEventData(mv_events=[seq]), num_steps=2, rng=0)
    ref, ref_times, _ = model.sample(seq, num_steps=2, rng=0)
    assert torch.equal(out.time_points, ref.time_points) and len(times) == len(ref_times) == 2
    with pytest.raises(ValueError, match="one sequence"):
        model.sample(BatchedMVEventData(mv_events=[seq, seq]), num_steps=1)


def test_relative_save_path_is_relative_to_the_working_directory(tmp_path, monkeypatch):
    from pathlib import Path
    from torchpp.training.config import SaveConfig
    from torchpp.training.training_helpers import resolve_save_path

    monkeypatch.chdir(tmp_path)
    path = resolve_save_path(SaveConfig(path="runs/m.pth"), Path("/elsewhere"), "run", "benchmark")
    assert path == tmp_path / "runs" / "m.pth"
