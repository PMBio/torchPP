"""One scored-event set for training, validation and evaluation; truncation keeps a valid window."""

import warnings

import numpy as np
import pytest
import torch
from torchpp.utils.event_data import TokenRoles

from torchpp import BatchedMVEventData, MVEventData, create_model
from torchpp.training.trainer import _aggregate_val_ll
from torchpp.training.training_helpers import make_collate_fns
from torchpp.training.config import BatchingConfig
from torchpp.utils.batching import TruncationStrategy, check_truncation
from torchpp.utils.evaluation import _build_prediction_items, evaluate_tpp


def _seq(times, types, T_obs=None):
    return MVEventData(
        torch.tensor(times, dtype=torch.float64), torch.tensor(types, dtype=torch.long), T_obs=T_obs
    )


SEQ = _seq([0.3, 0.3, 0.9, 1.4, 2.0], [0, 1, 1, 0, 1], T_obs=2.5)


# ---------------------------------------------------------------------------
# Scored events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "has_origin, context, expected",
    [(False, None, 3), (True, None, 5), (False, [1], 1), (True, [1], 2)],
)
def test_num_scored_events(has_origin, context, expected):
    model = create_model("poisson", D=2, token_roles=TokenRoles(has_origin=has_origin, context_types=context or ()))
    assert model.num_scored_events(SEQ).tolist() == [expected]
    assert model.num_scored_events(SEQ.to_batch()).tolist() == model._scored_mask(SEQ.to_batch()).sum(1).tolist()


@pytest.mark.parametrize("has_origin", [False, True])
@pytest.mark.parametrize("context", [None, [1]])
def test_eval_targets_equal_the_scored_set(has_origin, context):
    model = create_model("poisson", D=2, token_roles=TokenRoles(has_origin=has_origin, context_types=context or ()))
    scored = torch.nonzero(model._scored_mask(SEQ.to_batch())[0]).flatten().tolist()
    items = _build_prediction_items([SEQ], has_origin=has_origin, predictable_mask=model.predictable_mask)
    assert sorted(it["history_length"] for it in items) == scored


def test_validation_ll_is_per_scored_event():
    model = create_model("poisson", D=2, seed=0).double()
    batch = BatchedMVEventData(mv_events=[SEQ, _seq([0.5, 1.0], [0, 0], T_obs=1.5)])
    val_ll, _ = _aggregate_val_ll(model, [batch], "cpu")
    with torch.no_grad():
        expected = float(model.likelihood(batch).sum() / model.num_scored_events(batch).sum())
    assert val_ll == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_origin", [False, True])
def test_evaluation_decomposes_the_full_likelihood(has_origin):
    """Σ next-event joint LL + censored tails = the full-sequence LL that training optimises."""
    model = create_model("const_exp_hawkes", D=2, seed=0, token_roles=TokenRoles(has_origin=has_origin)).double().eval()
    seqs = [SEQ, _seq([0.4], [1], T_obs=1.0), _seq([0.2, 0.7], [0, 1], T_obs=0.7)]
    m = evaluate_tpp(model, seqs, compute_marginals=False)
    with torch.no_grad():
        tails = sum(
            float(model.survival_time(torch.tensor([s.T_obs], dtype=torch.float64), MVEventData(s.time_points, s.event_types), log=True))
            for s in seqs
        )
        full = float(model.likelihood(BatchedMVEventData(mv_events=seqs)).sum())
    assert m["num_predictions"] == int(model.num_scored_events(BatchedMVEventData(mv_events=seqs)).sum())
    assert m["joint_ll"] * m["num_predictions"] + tails == pytest.approx(full, rel=1e-9)
    assert m["ll_per_event"] == pytest.approx(full / m["num_scored_events"], rel=1e-12)


def test_evaluation_with_an_origin_scores_single_event_sequences():
    model = create_model("poisson", D=2, seed=0, token_roles=TokenRoles(has_origin=True)).double().eval()
    seqs = [_seq([0.4], [1], T_obs=1.0), _seq([0.9], [0], T_obs=3.0)]
    m = evaluate_tpp(model, seqs, compute_marginals=False)
    assert m["num_predictions"] == 2 and m["num_scored_events"] == 2


def test_evaluation_accepts_censored_sequences_without_events():
    model = create_model("poisson", D=2, seed=0, token_roles=TokenRoles(has_origin=True)).double().eval()
    empty = MVEventData(torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long), T_obs=2.0)
    m = evaluate_tpp(model, [empty], compute_marginals=False)
    with torch.no_grad():
        expected = float(model.likelihood(empty.to_batch()))
    assert m["num_predictions"] == 0 and m["num_scored_events"] == 0
    assert m["ll_per_sequence"] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(m["ll_per_event"])


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

LONG = _seq([0.2, 0.5, 0.9, 1.3, 1.8, 2.4], [0, 1, 0, 1, 1, 0], T_obs=3.0)


def test_right_truncation_ends_the_window_at_the_last_kept_event():
    cut = TruncationStrategy.right_truncate(LONG, 3)
    assert cut.time_points.tolist() == [0.2, 0.5, 0.9] and cut.T_obs == pytest.approx(0.9)


def test_left_truncation_keeps_T_obs():
    cut = TruncationStrategy.left_truncate(LONG, 3)
    assert cut.time_points.tolist() == [1.3, 1.8, 2.4] and cut.T_obs == pytest.approx(3.0)


def test_random_truncation_is_a_contiguous_stretch():
    for seed in range(10):
        cut = TruncationStrategy.random_sample(LONG, 3, seed)
        idx = [LONG.time_points.tolist().index(t) for t in cut.time_points.tolist()]
        assert idx == list(range(idx[0], idx[0] + 3))
        assert cut.T_obs == pytest.approx(3.0 if idx[-1] == len(LONG) - 1 else float(LONG.time_points[idx[-1]]))
    same = [TruncationStrategy.random_sample(LONG, 3, 7).time_points.tolist() for _ in range(2)]
    assert same[0] == same[1]


def test_left_truncated_window_scores_like_a_sequence_starting_at_its_first_event():
    """Without an origin a mid-sequence stretch is conditioned on its first event (Poisson closed form)."""
    model = create_model("poisson", D=2, seed=0).double()
    cut = TruncationStrategy.left_truncate(LONG, 3)
    with torch.no_grad():
        ll = float(model.likelihood(cut.to_batch()))
        mu = model.intensity(torch.tensor([cut.T_obs], dtype=torch.float64), cut)[0]
    expected = float(torch.log(mu[cut.event_types[1:]]).sum() - mu.sum() * (cut.T_obs - float(cut.time_points[0])))
    assert ll == pytest.approx(expected, rel=1e-12)


def test_truncation_warns_when_an_origin_is_lost():
    for strategy, warns in (("left", True), ("random", True), ("right", False)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_truncation(strategy, has_origin=True)
            check_truncation(strategy, has_origin=False)
        assert (len(w) == 1) is warns
    batching = BatchingConfig(batch_size=2, max_sequence_length=3, truncation_strategy="left")
    with pytest.warns(UserWarning, match="origin"):
        make_collate_fns(batching, seed=0, has_origin=True)


def test_prediction_items_follow_the_sequence_device():
    """The predictable mask is moved to the sequences' device (it may come from a model on another device)."""
    seq = MVEventData(torch.tensor([0.2, 0.5, 0.9]), torch.tensor([0, 1, 0]))
    items = _build_prediction_items([seq], predictable_mask=torch.ones(2, dtype=torch.bool))
    assert sorted(item["history_length"] for item in items) == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_evaluate_tpp_with_sequences_on_the_gpu():
    model = create_model("const_exp_hawkes", 2, seed=0).to("cuda")
    seqs = [MVEventData(torch.tensor([0.2, 0.5, 0.9, 1.3]), torch.tensor([0, 1, 0, 1])).to("cuda")] * 3
    metrics = evaluate_tpp(model, seqs, device="cuda", compute_marginals=False)
    assert np.isfinite(metrics["joint_ll"]) and np.isfinite(metrics["ll_per_event"])
    items = _build_prediction_items(seqs, predictable_mask=torch.ones(2, dtype=torch.bool))
    assert len(items) == 9
