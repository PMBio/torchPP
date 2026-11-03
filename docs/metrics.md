# Evaluation metrics

`evaluate_tpp` is used by `train.py` after training and by `evaluate.py`. It reports two kinds
of numbers on the same set of events, the scored events of the likelihood (see
[Notation → Scored events](./notation.md#scored-events)).

- `ll_per_event` is the full-sequence log-likelihood, including the censored tail up to
  `T_obs`, divided by the number of scored events. It is the training objective and the
  validation number.
- The next-event predictions have one target per scored event, predicted from the events
  before it. With an origin (`has_origin=True`), the first event is a target too, and its wait
  is measured from 0. Without an origin, which is common for benchmarks with an arbitrary time
  origin, the first event and the events tied with it are history only. Context types and
  ignored types (`TokenRoles.ignored_types`) are never targets.

Sequences with a single event, or with no events (censored only), are evaluated as well. They
contribute to `ll_per_event` through their survival term.

All metrics are means over targets, in the native time units of the data. The notation follows
[Notation](./notation.md).

| Key | Meaning |
|---|---|
| `ll_per_event` | full-sequence log-likelihood per scored event (events plus censored tail) |
| `ll_per_sequence` | the same log-likelihood averaged over sequences |
| `num_scored_events`, `num_sequences` | the counts behind the two averages |
| `joint_ll` | $\log p(t_i, m_i \mid \mathcal{H})$, the joint log-density of the next event |
| `marginal_time_ll` | $\log p(t_i \mid \mathcal{H})$ |
| `cond_type_ll` | $\log p(m_i \mid t_i, \mathcal{H})$, with `joint_ll = marginal_time_ll + cond_type_ll` |
| `cond_top1_accuracy`, `cond_top5_accuracy` | type accuracy from $p(m \mid t_i, \mathcal{H})$, the type at the true time (Acc\|t) |
| `marginal_type_ll` | $\log p(m_i \mid \mathcal{H})$, type without knowing the time |
| `top1_accuracy`, `top5_accuracy` | type accuracy from $p(m \mid \mathcal{H})$ (marginal Acc) |
| `time_rmse_mean`, `time_mae_mean` | error of the predictive mean $E[t_i \mid \mathcal{H}]$ |
| `time_rmse_median`, `time_mae_median` | error of the predictive median |
| `num_predictions` | number of targets |
| `num_floored_intensity` | targets whose intensity underflowed and was floored inside `log` (0 means none) |

- The marginal metrics (`marginal_type_ll`, `top*_accuracy`, RMSE/MAE) integrate over the
  next-event time on a grid and are the expensive part of the evaluation.
  `eval.compute_marginal_type=false` skips them. Targets with an infinite mean or median (the
  event may never occur) are left out of RMSE/MAE.
- Top-5 accuracy is 1 by construction when $D \le 5$.
- `default_fit` and `default_eval` evaluate only the first 1000 test sequences, and the
  printout says when a subset was used. Use `-c eval/full` (all test sequences) for reported
  numbers.

## Validation log-likelihood during training

`train.py` prints `Val LL (per event)`. This is the full-sequence log-likelihood (positive term
and survival integral), summed over the validation set and divided by its number of scored
events. It selects the best checkpoint and drives early stopping, and it is the same quantity as
the test `ll_per_event`. The next-event `joint_ll` differs from it by the censored tails. Summed
over targets, `joint_ll` plus the tails gives the full log-likelihood.
