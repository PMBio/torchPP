"""Independent global closed-form CI oracles for the CI-equivalence tests.

The equivalence check ``Σ _interval_ci ≡ global closed form`` needs a global CI that is
computed independently of the per-interval path:

* Baseline / Hawkes families: their live ``_closed_form_ci`` already IS the global
  module CI (an independent global form), so it serves as its own oracle.
* Neural closed-form heads have no independent global ``_closed_form_ci`` override
  (they use the base ``Σ _interval_ci``).  Their independent oracle is the exact
  vectorised per-interval-layout CI, implemented here on top of the model's public
  methods.

Importable from any test subdir via ``pythonpath = tests``.
"""

from __future__ import annotations

import torch

from torchpp.tpp.base_tpp import IntensityBasedTPP
from torchpp.utils.event_data import BatchedMVEventData


def neural_layout_ci(m, effective_T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
    """Vectorised piecewise CI over all B×(L+1)×D intervals — relocated neural ``_raw_ci``.

    Verbatim relocation of the deleted neural ``_raw_ci`` + ``_ci_interval_layout``,
    calling the model's still-live methods via ``m.``.  ``effective_T`` is ``(B, D)``
    (per-type upper limit, already clamped to termination); returns ``(B, D)`` identical
    to what the deleted neural ``_raw_ci(effective_T, ts)`` returned.

    Interval layout (one row in full_params per interval):
        slot 0      : [0,         tp[:,0]]   governed by prior  (col 0)
        slot l∈[1,L): [tp[:,l-1], tp[:,l]]   governed by col l  (causal)
        slot L      : [tp_last,   eff_T]     governed by col after last real event
    """
    full_params = m._full_params_cached(ts)

    B = effective_T.shape[0]
    L = full_params.shape[1] - 1  # number of events per batch element (some might be padded)
    dtype = effective_T.dtype
    device = effective_T.device

    tp = ts.time_points.to(dtype)  # (B, L)
    real_mask = ts.event_mask  # (B, L)

    # Causal index selects the params of the event causally preceding each position,
    # so simultaneous events do not observe each other's contribution.
    col_idx = ts.causal_indices  # (B, L)
    trailing = full_params.shape[2:]
    idx_events = col_idx.view(B, L, *([1] * len(trailing))).expand(B, L, *trailing)
    params_events = full_params.gather(1, idx_events).to(dtype=dtype)  # (B, L, ...)

    # Params after the last real event for the tail interval [tp[:,last], effective_T].
    real_counts = ts.real_counts  # (B,)
    idx_final = real_counts.view(B, 1, *([1] * len(trailing))).expand(B, 1, *trailing)
    params_final = full_params.gather(1, idx_final).to(dtype=dtype)  # (B, 1, ...)

    params_all = torch.cat([params_events, params_final], dim=1)  # (B, L+1, ...)

    # Interval bounds. col_idx[b, l] = index into tp_with_zero of the last real event's
    # time (0 for no prior events); prepend a zero column so col_idx == 0 maps to 0.0.
    tp_with_zero = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), tp], dim=1)  # (B, L+1)
    t_lo_events = tp_with_zero.gather(1, col_idx)  # (B, L)
    t_lo_final = tp_with_zero.gather(1, real_counts.unsqueeze(1)).squeeze(1)  # (B,)

    t_lo_all = (
        torch.cat([t_lo_events, t_lo_final.unsqueeze(1)], dim=1).unsqueeze(-1).expand(B, L + 1, m.D)
    )  # (B, L+1, D)

    # t_hi for 0..L-1: min(tp[l], effective_T) per type; tail interval ends at effective_T.
    t_hi_events = torch.minimum(
        tp.unsqueeze(-1).expand(B, L, m.D),
        effective_T.unsqueeze(1).expand(B, L, m.D),
    )  # (B, L, D)
    t_hi_final = effective_T.unsqueeze(1)  # (B, 1, D)

    t_hi_all = torch.cat([t_hi_events, t_hi_final], dim=1)  # (B, L+1, D)

    # Slots 0..L-1 are active only where a real event occupies the position; the final
    # slot is always active.  Padded events always follow all real events.
    real_mask_ext = torch.cat(
        [real_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1
    )  # (B, L+1)

    # Deduplicate simultaneous events: positions sharing a causal column produce identical
    # intervals with identical params.  Keep only the first per causal column (col_idx is
    # non-decreasing, so ties are consecutive); the final slot (index L) is always kept.
    first_occurrence = torch.ones(B, L + 1, dtype=torch.bool, device=device)
    if L > 0:
        first_occurrence[:, 1:L] = col_idx[:, 1:] != col_idx[:, :-1]

    # Exclude intervals where effective_T lies below the interval's lower bound (can happen
    # when effective_T is clamped by termination).
    needs_int = (
        real_mask_ext.unsqueeze(-1)  # (B, L+1, 1)
        & first_occurrence.unsqueeze(-1)  # (B, L+1, 1)
        & (t_hi_all > t_lo_all)  # (B, L+1, D)
    )  # (B, L+1, D)

    # Clamp invalid intervals to zero length to avoid negative-interval numerics.
    t_hi_all = torch.where(needs_int, t_hi_all, t_lo_all)  # (B, L+1, D)

    integrals = m._interval_ci_from_params(params_all, t_lo_all, t_hi_all)  # (B, L+1, D)
    ci = torch.where(needs_int, integrals, torch.zeros_like(integrals)).sum(dim=1)  # (B, D)
    return ci


def global_ci_oracle(m, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
    """Independent global closed-form CI Λ_d(T) — the ``Σ _interval_ci`` equivalence oracle.

    ``T`` is ``(B,)``; returns ``(B, D)``.  Termination folds into ``_effective_T``
    (still live on the model).
    """
    eff = m._effective_T(T, ts)  # (B, D)
    if type(m)._closed_form_ci is IntensityBasedTPP._closed_form_ci:
        # neural closed-form: no independent global override -> relocated vectorised-layout oracle
        return neural_layout_ci(m, eff, ts)
    # baseline / hawkes: their live _closed_form_ci IS the independent global form
    return m._closed_form_ci(ts, None, T)
