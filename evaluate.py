"""Standalone evaluation entry point for trained TPP models.

Mirrors ``train.py``: dataset selection is driven entirely by the
``dataset_type`` field in the config and the registered dataset loaders, so
the same script evaluates UKB, EasyTPP, MIMIC-IV, or any future registered
dataset.  The ``-c data/<preset>`` YAML files used for training apply here
unchanged.

Usage examples
--------------
.. code-block:: bash

    # UKB
    python evaluate.py -c default_eval -c data/ukb_real_full \\
        model_path=models/my_model.pth model_type=spline_exp_hawkes

    # EasyTPP — retweet
    python evaluate.py -c default_eval -c data/easytpp_retweet \\
        model_path=models/retweet_neural_thp.pth model_type=neural_thp
"""

import json
import math
import os
from pathlib import Path

import numpy as np

# Importing the data package triggers @register_dataset decorators for all
# built-in dataset classes.
import torchpp.data  # noqa: F401
from torchpp.training.config import StandaloneEvalConfig
from torchpp.training.config_utils import load_config, print_config
from torchpp.training.dataset_registry import get_data_root
from torchpp.training.model_factory import create_model
from torchpp.training.training_helpers import load_test_dataset, truncate_sequences_for_eval
from torchpp.utils.evaluation import evaluate_tpp, format_metrics


def main() -> None:
    config: StandaloneEvalConfig = load_config(StandaloneEvalConfig)
    print_config(config, title="Evaluation Configuration")

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    DATA_ROOT = get_data_root()

    test_dataset = load_test_dataset(
        config.dataset_type,
        dict(config.dataset),
        DATA_ROOT,
        limit_test_size=config.eval.limit_test_size,
    )
    D = test_dataset.num_event_types
    print(f"  {D} event types, terminating={test_dataset.terminating}")

    model, _ = create_model(
        model_type=config.model_type,
        D=D,
        arch=config.arch,
        terminating=test_dataset.terminating,
        load_path=model_path,
    )
    model = model.to(config.device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Replay the training truncation policy (e.g. MIMIC 316) so standalone RMSE/Acc/p(d)
    # match the in-training test eval; no-op when max_sequence_length is None.
    eval_sequences = test_dataset
    if config.eval.max_sequence_length is not None:
        print(
            "Applying truncation policy to test split: "
            f"max_length={config.eval.max_sequence_length}, strategy='{config.eval.truncation_strategy}'"
        )
        eval_sequences = truncate_sequences_for_eval(
            list(test_dataset),
            max_length=config.eval.max_sequence_length,
            strategy=config.eval.truncation_strategy,
            seed=config.arch.seed,
        )

    # Opt-in per-item dump (subset analyses, e.g. paper/binned/mimic_tie_split.py): TORCHPP_EVAL_DUMP_ITEMS=<file.npz>.
    dump_path = os.environ.get("TORCHPP_EVAL_DUMP_ITEMS")

    print("\nEvaluating model on test set…")
    metrics = evaluate_tpp(
        model=model,
        sequences=eval_sequences,
        num_event_types=D,
        device=config.device,
        eval_batch_size=config.eval.eval_batch_size,
        ignore_indices=test_dataset.ignore_indices,
        compute_marginals=config.eval.compute_marginal_type,
        integrate_marginal_via_grid=config.eval.integrate_marginal_via_grid,
        grid_points_per_unit_time=config.eval.grid_points_per_unit_time,
        time_scale_factor=test_dataset.time_scale_factor,
        return_items=bool(dump_path),
    )
    if dump_path:
        items = metrics.pop("items")
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, time_scale_factor=test_dataset.time_scale_factor, **items)
        print(f"Per-item dump written to {dump_path}")

    print("\n" + "=" * 60)
    print("Test Evaluation Results:")
    print("=" * 60)
    print(format_metrics(metrics, prefix="  "))
    print("=" * 60)

    # Machine-readable, self-describing result line (safe to collect from the job logs:
    # grep the tag, json.loads the rest — no prose parsing, no filename guessing).
    # Non-finite floats -> null so it is valid JSON for any parser.
    record: dict = {
        "model_type": config.model_type,
        "dataset": config.dataset.get("dataset_name", config.dataset_type),
        "model_path": str(config.model_path),
        "seed": config.arch.seed,
        "grid_points_per_unit_time": config.eval.grid_points_per_unit_time,
        "limit_test_size": config.eval.limit_test_size,
        "max_sequence_length": config.eval.max_sequence_length,
        "n_event_types": D,
        "n_sequences": len(eval_sequences),
    }
    for k, v in metrics.items():
        record[k] = None if (isinstance(v, float) and not math.isfinite(v)) else v
    print("EVAL_METRICS_JSON " + json.dumps(record))


if __name__ == "__main__":
    main()
