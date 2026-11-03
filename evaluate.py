"""Standalone evaluation entry point for trained TPP models.

A checkpoint written by ``train.py`` (or :func:`torchpp.training.save_model`) records its
model type, architecture and dataset, so ``model_path`` is enough. Explicit command-line
values override the stored ones: a data preset or ``dataset.*`` selects another dataset,
``arch.*=...`` changes an architecture field (e.g. a finer integration grid). A file that
holds only a ``state_dict`` needs ``model_type`` and the architecture given explicitly.

Usage examples
--------------
.. code-block:: bash

    python evaluate.py -c default_eval model_path=models/benchmark/taxi/thp.pth

    # bare state_dict
    python evaluate.py -c default_eval -c data/retweet -c model/thp \\
        model_path=models/retweet_thp.pth model_type=thp
"""

import json
import sys
from pathlib import Path

from omegaconf.errors import MissingMandatoryValue

# Importing the data package triggers @register_dataset decorators for all
# built-in dataset classes.
import torchpp.data  # noqa: F401
from torchpp.training.arch import build_arch, merge_arch
from torchpp.training.checkpoint import _token_roles, arch_from_dict, read_checkpoint
from torchpp.training.config import StandaloneEvalConfig
from torchpp.training.config_utils import _split_argv, load_config, print_config
from torchpp.training.dataset_registry import get_data_root
from torchpp.training.model_factory import create_model
from torchpp.training.training_helpers import load_test_dataset
from torchpp.utils.evaluation import evaluate_tpp, format_metrics


def _cli_keys(argv: list[str]) -> set[str]:
    """Dotted keys given explicitly as ``key=value`` on the command line."""
    _, overrides, _, _ = _split_argv(argv)
    return {o.split("=", 1)[0] for o in overrides}


def main() -> None:
    argv = sys.argv[1:]
    try:
        run(argv)
    except MissingMandatoryValue as err:
        key = getattr(err, "full_key", "") or str(err)
        hint = " Add -c model/<type>." if "model_type" in key else ""
        sys.exit(f"error: missing required value '{key}' (set {key}=...).{hint}")
    except (ValueError, FileNotFoundError, RuntimeError) as err:
        sys.exit(f"error: {err}")


def run(argv: list[str]) -> None:
    config: StandaloneEvalConfig = load_config(
        StandaloneEvalConfig,
        argv=argv,
        configs_dir=Path(__file__).resolve().parent / "configs",
        description="Evaluate a saved checkpoint on a test set (model_path=<file>).",
    )
    print_config(config, title="Evaluation Configuration")

    model_path = Path(config.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path={model_path} does not exist.")
    ckpt = read_checkpoint(model_path)

    if "model_type" in ckpt:
        model_type = config.model_type or ckpt["model_type"]
        # Only arch.*=... given on the command line change the checkpoint's architecture.
        overrides: dict = {}
        for key in _cli_keys(argv):
            if key.startswith("arch."):
                path, node, src = key.split(".")[1:], overrides, config.arch
                for part in path[:-1]:
                    node, src = node.setdefault(part, {}), src[part]
                node[path[-1]] = src[path[-1]]
        arch = merge_arch(model_type, arch_from_dict(model_type, ckpt.get("arch")), overrides)
        if config.dataset:
            dataset_type, dataset_cfg = config.dataset_type, dict(config.dataset)
        else:
            dataset_type = ckpt["dataset_type"] or config.dataset_type
            dataset_cfg = dict(ckpt["dataset"] or {})
    else:
        if config.model_type is None:
            raise ValueError(
                f"{model_path} holds only a state_dict: pass model_type=... and the architecture "
                "(e.g. -c model/<type>) used in training."
            )
        model_type, arch = config.model_type, build_arch(config.model_type, config.arch)
        dataset_type, dataset_cfg = config.dataset_type, dict(config.dataset)

    test_dataset = load_test_dataset(
        dataset_type,
        dataset_cfg,
        get_data_root(),
        limit_test_size=config.eval.limit_test_size,
    )
    D = test_dataset.num_event_types
    if "D" in ckpt and ckpt["D"] != D:
        raise ValueError(f"{model_path} was trained with D={ckpt['D']} event types; the test set has {D}.")
    terminating, roles = test_dataset.terminating, test_dataset.token_roles
    saved = {"terminating": ckpt.get("terminating")}
    if "token_roles" in ckpt or "has_origin" in ckpt:
        saved["has_origin"] = _token_roles(ckpt).has_origin
    for key, data_value in (("terminating", terminating), ("has_origin", roles.has_origin)):
        if saved.get(key) is not None and bool(saved[key]) != bool(data_value):
            raise ValueError(
                f"{model_path} was trained with {key}={saved[key]}; the test set has {key}={data_value}."
            )
    print(f"  {D} event types, terminating={terminating}")

    model = create_model(
        model_type=model_type,
        D=D,
        arch=arch,
        terminating=terminating,
        load_path=model_path,
        token_roles=roles,
        verbose=True,
    )
    model = model.to(config.device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    print("\nEvaluating model on test set…")
    metrics = evaluate_tpp(
        model=model,
        sequences=test_dataset,
        num_event_types=D,
        device=config.device,
        eval_batch_size=config.eval.eval_batch_size,
        compute_marginals=config.eval.compute_marginal_type,
        integrate_marginal_via_grid=config.eval.integrate_marginal_via_grid,
        grid_points_per_unit_time=config.eval.grid_points_per_unit_time,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Test Evaluation Results:")
    print("=" * 60)
    print(format_metrics(metrics, prefix="  "))
    print("=" * 60)

    out = Path(config.output_path) if config.output_path else model_path.with_suffix(".eval.json")
    out.write_text(json.dumps(
        {
            "model_path": str(model_path),
            "model_type": model_type,
            "dataset_type": dataset_type,
            "dataset": dataset_cfg,
            "n_test_sequences": len(test_dataset),
            "metrics": metrics,
        },
        indent=2,
        default=float,
    ))
    print(f"Metrics written to {out}")


if __name__ == "__main__":
    main()
