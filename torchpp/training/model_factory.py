"""Single authoritative entry point for instantiating any supported TPP model
from a ``model_type`` string and an architecture config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from torchpp.tpp.base_tpp import TemporalPointProcess
from torchpp.training.config import ModelArchConfig
from torchpp.utils.event_data import DatasetStats
from torchpp.utils.math import interaction_spline_knot_gaps
from torchpp.tpp.baseline import (
    PoissonProcess,
    ConditionalInhomogeniousPoissonProcess,
    SplinePoissonProcess,
    PiecewiseConstantPoissonProcess,
)
from torchpp.tpp.hawkes import (
    ConstBaselineExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    InhibitiveConstExpHawkesProcess,
    InhibitiveSplineExpHawkesProcess,
    ConstBaselineSplineKernelHawkesProcess,
    SplineBaselineSplineKernelHawkesProcess,
    SplineBaselineNSSplineKernelHawkesProcess,
    ConstBaselineNSExpKernelHawkesProcess,
    SplineBaselineNSExpKernelHawkesProcess,
    InhibitiveSplineBaselineSplineKernelHawkesProcess,
    SplineBaselineNeuralKernelHawkesProcess,
    SplineBaselineNeuralSplineKernelHawkesProcess,
    NumericalSplineBaselineExpKernelHawkesProcess,
    PiecewiseBaselineExpKernelHawkesProcess,
    ConstBaselinePiecewiseKernelHawkesProcess,
)
from torchpp.tpp.neural import (
    NeuralPoissonProcess,
    NeuralLogLinearProcess,
    NeuralSplineProcess,
    NeuralTHPProcess,
    NeuralSAHPProcess,
    EasyTPPTHPProcess,
    EasyTPPSAHPProcess,
    NeuralMLPProcess,
    NeuralPiecewiseConstantProcess,
    IntensityFreeProcess,
    NHPProcess,
    AttNHPProcess,
)
from torchpp.modules.nn.transformer import TransformerConfig, TransformerEncoder
from torchpp.modules.nn.encoder import GRUEncoder, ContTimeLSTMEncoder
from torchpp.modules.nn.attnhp import AttNHPEncoder

_MODEL_DESCRIPTIONS: dict[str, str] = {
    "poisson": "Constant Poisson Process",
    "inhomogeneous_poisson": "Log-Linear Poisson Process",
    "spline_poisson": "Spline Poisson Process",
    "piecewise_constant": "Piecewise Constant Poisson Process",
    "hawkes": "Const-Baseline Exp-Kernel Hawkes Process",
    "linear_exp_hawkes": "Linear-Baseline Exp-Kernel Hawkes Process",
    "spline_exp_hawkes": "Spline-Baseline Exp-Kernel Hawkes Process",
    "const_spline_kernel": "Const-Baseline Spline-Kernel Hawkes Process",
    "spline_spline_kernel": "Spline-Baseline Spline-Kernel Hawkes Process",
    "spline_ns_spline_kernel": "Spline-Baseline Non-Stationary-Spline-Kernel Hawkes Process",
    "ns_exp_hawkes": "Const-Baseline Non-Stationary-Exp-Kernel Hawkes Process",
    "spline_ns_exp_hawkes": "Spline-Baseline Non-Stationary-Exp-Kernel Hawkes Process",
    "numerical_spline_exp_hawkes": "Numerical Spline-Baseline Exp-Kernel Hawkes Process",
    "piecewise_exp_hawkes": "Piecewise-Baseline Exp-Kernel Hawkes Process",
    "const_piecewise_kernel": "Const-Baseline Piecewise-Kernel Hawkes Process",
    "inhibitive_const_exp_hawkes": "Inhibitive Const-Baseline Exp-Kernel Hawkes Process",
    "inhibitive_spline_exp_hawkes": "Inhibitive Spline-Baseline Exp-Kernel Hawkes Process",
    "inhibitive_spline_spline_kernel": "Inhibitive Spline-Baseline Spline-Kernel Hawkes Process",
    "neural_hawkes": "Spline-Baseline Neural-Kernel Hawkes Process",
    "neural_spline_hawkes": "Spline-Baseline Neural-Spline-Kernel Hawkes Process",
    "neural_poisson": "Neural Poisson Process",
    "neural_log_linear": "Neural Log-Linear Process",
    "neural_spline": "Neural Spline Process",
    "neural_mlp": "Neural MLP Intensity Process",
    "neural_piecewise_constant": "Neural Piecewise Constant Process",
    "neural_thp": "Neural Transformer Hawkes Process (THP)",
    "neural_sahp": "Neural Self-Attentive Hawkes Process (SAHP)",
    "easytpp_thp": "EasyTPP-replica Transformer Hawkes Process (THP)",
    "easytpp_sahp": "EasyTPP-replica Self-Attentive Hawkes Process (SAHP)",
    "intensity_free": "IntensityFree (log-normal mixture, Shchur et al. 2020)",
    "nhp": "Neural Hawkes Process (continuous-time LSTM, Mei & Eisner 2017)",
    "attnhp": "Attentive Neural Hawkes Process (Yang et al. 2022)",
}


def _build_run_name(model_type: str, arch: ModelArchConfig) -> str:
    """Generate a descriptive run/save name from model_type + arch config.

    Naming convention:
        Poisson:     const_poisson, loglinear_poisson, spline_K5_poisson, piecewise_K5_poisson
        Hawkes:      const_exp_hawkes, linear_exp_hawkes, spline_K5_exp_hawkes,
                     const_spline_kK5_hawkes, spline_K5_spline_kK5_hawkes
        Inhibitive:  inhibitive_relu_const_exp_hawkes, inhibitive_softplus_const_exp_hawkes,
                     + _constrained suffix when constrain_kernel=True
        Neural:      neural_4l_128d_poisson, neural_4l_128d_spline_K5
    """
    K = arch.spline_K
    kK = arch.spline_kernel_K
    clamp = arch.inhibitive_clamping_fn

    # --- Poisson family ---
    if model_type == "poisson":
        return "const_poisson"
    if model_type == "inhomogeneous_poisson":
        return "loglinear_poisson"
    if model_type == "spline_poisson":
        return f"spline_K{K}_poisson"
    if model_type == "piecewise_constant":
        return f"piecewise_K{K}_poisson"

    # --- Hawkes family (baseline_kernel_hawkes) ---
    if model_type == "hawkes":
        return "const_exp_hawkes"
    if model_type == "linear_exp_hawkes":
        return "linear_exp_hawkes"
    if model_type == "spline_exp_hawkes":
        return f"spline_K{K}_exp_hawkes"
    if model_type == "piecewise_exp_hawkes":
        return f"piecewise_K{K}_exp_hawkes"
    if model_type == "numerical_spline_exp_hawkes":
        return f"numerical_spline_K{K}_exp_hawkes"
    if model_type == "const_spline_kernel":
        return f"const_spline_kK{kK}_hawkes"
    if model_type == "const_piecewise_kernel":
        return f"const_piecewise_kK{kK}_hawkes"
    if model_type == "spline_spline_kernel":
        return f"spline_K{K}_spline_kK{kK}_hawkes"
    if model_type == "spline_ns_spline_kernel":
        return f"spline_K{K}_ns_K{arch.spline_ns_K}_spline_kK{kK}_hawkes"
    if model_type == "ns_exp_hawkes":
        return f"const_ns_K{arch.spline_ns_K}_exp_hawkes"
    if model_type == "spline_ns_exp_hawkes":
        return f"spline_K{K}_ns_K{arch.spline_ns_K}_exp_hawkes"

    # --- Inhibitive family ---
    if model_type == "inhibitive_const_exp_hawkes":
        name = f"inhibitive_{clamp}_const_exp_hawkes"
        if arch.constrain_kernel:
            name += "_constrained"
        return name
    if model_type == "inhibitive_spline_exp_hawkes":
        return f"inhibitive_{clamp}_spline_K{K}_exp_hawkes"
    if model_type == "inhibitive_spline_spline_kernel":
        return f"inhibitive_{clamp}_spline_K{K}_spline_kK{kK}_hawkes"
    if model_type == "neural_hawkes":
        return f"spline_K{K}_neural_kK{kK}_hawkes"
    if model_type == "neural_spline_hawkes":
        return f"spline_K{K}_neural_spline_kK{kK}_hawkes"

    # --- Neural family (arch_head) ---
    if model_type.startswith("neural_"):
        arch_tag = f"neural_{arch.n_layer}l_{arch.n_embd}d"
        head = model_type.removeprefix("neural_")  # poisson, log_linear, spline, thp, sahp, mlp
        if head == "spline":
            return f"{arch_tag}_{head}_kK{kK}"
        return f"{arch_tag}_{head}"

    # --- EasyTPP-replica family ---
    if model_type.startswith("easytpp_"):
        arch_tag = f"easytpp_{arch.n_layer}l_{arch.n_embd}d"
        head = model_type.removeprefix("easytpp_")  # thp, sahp
        return f"{arch_tag}_{head}"

    return model_type


def create_model(
    model_type: str,
    D: int,
    arch: ModelArchConfig,
    terminating: bool = True,
    seed: Optional[int] = None,
    load_path: Optional[Path] = None,
    dataset_stats: Optional[DatasetStats] = None,
) -> tuple[TemporalPointProcess, str]:
    """Instantiate a TPP model by name.

    Args:
        model_type:     One of the strings in ``VALID_MODEL_TYPES``.
        D:              Number of event types (dimension).
        arch:           Architecture hyper-parameters (spline knots, CI steps, …).
        terminating:    Whether each event type can occur at most once.
        seed:           Optional RNG seed for parameter initialisation.
                        Pass ``None`` when loading weights from a checkpoint.
        load_path:      If given, load ``state_dict`` from this path after
                        constructing the model.
        dataset_stats:  Optional empirical statistics computed from the training
                        sequences.  When provided, Hawkes kernel/baseline modules
                        use data-driven initialisation (beta calibrated to observed
                        inter-event times, alpha seeded from pair-occurrence counts).
                        Pass ``None`` to use the original hardcoded ranges.

    Returns:
        ``(model, run_name)`` — the constructed model and a descriptive
        run name encoding model type and arch params (e.g.
        ``"spline_K5_exp_hawkes"``).  Used as the default save filename
        and wandb run name.

    Raises:
        ValueError: For unknown ``model_type``.
    """
    if model_type not in _MODEL_DESCRIPTIONS:
        from torchpp.training.config import VALID_MODEL_TYPES

        raise ValueError(f"Unknown model_type '{model_type}'. Valid choices: {', '.join(VALID_MODEL_TYPES)}")

    # Neural encoders/heads init from the GLOBAL torch RNG at construction, so pin it when a seed is
    # given, for reproducible weights regardless of ambient RNG state (the model's ``seed`` arg only
    # builds the sampling generator).  ``None`` = loading a checkpoint, so leave the RNG untouched.
    if seed is not None:
        torch.manual_seed(seed)

    description = _MODEL_DESCRIPTIONS[model_type]
    run_name = _build_run_name(model_type, arch)
    print(f"Creating {description} model (run_name={run_name})...")

    K = arch.spline_K
    kK = arch.spline_kernel_K
    ci = arch.num_ci_integration_steps

    # ── Data-driven spline time extents ──────────────────────────────────────
    # T_baseline: covers the observation window for baseline / piecewise models.
    #   Anchored to the 95th percentile of per-sequence censoring times so that
    #   virtually all sequences are within the spline range without being dominated
    #   by outliers.  Falls back to spline_T_factor itself when dataset_stats is
    #   unavailable (treats p95 = 1.0, so the factor is used as an absolute value).
    if dataset_stats is not None and dataset_stats.sequence_time_p95 > 0:
        T_baseline = arch.spline_T_factor * dataset_stats.sequence_time_p95
    else:
        T_baseline = arch.spline_T_factor

    # T_kernel: covers the decay time-scale for kernel / relative-time models.
    #   Neural spline/piecewise models are relative-time based (like kernels) —
    #   they predict intensity as a function of elapsed time since the last event,
    #   so their knots are positioned using interaction_spline_knot_locs(T_kernel, K)
    #   which applies the kernel schedule (spline_kernel_knot_schedule/power) at this scale.
    if dataset_stats is not None and dataset_stats.inter_event_time_p95 > 0:
        T_kernel = arch.spline_kernel_T_factor * dataset_stats.inter_event_time_p95
    else:
        T_kernel = arch.spline_kernel_T_factor

    dt = T_baseline / (K - 1)  # scalar spacing for baseline models
    kdt = interaction_spline_knot_gaps(
        T_kernel,
        arch.spline_kernel_K,
        schedule=arch.spline_kernel_knot_schedule,
        power=arch.spline_kernel_knot_power,
    )  # (kK-1,) gaps for Hawkes kernels
    neural_kdt = interaction_spline_knot_gaps(
        T_kernel,
        arch.spline_kernel_K,
        schedule=arch.spline_kernel_knot_schedule,
        power=arch.spline_kernel_knot_power,
    )  # (kK-1,) gaps for neural models — uses spline_kernel_K, not spline_K

    kw: dict = dict(terminating=terminating)
    if seed is not None:
        kw["seed"] = seed

    if model_type == "poisson":
        model = PoissonProcess(D=D, dataset_stats=dataset_stats, **kw)

    elif model_type == "inhomogeneous_poisson":
        model = ConditionalInhomogeniousPoissonProcess(D=D, dataset_stats=dataset_stats, **kw)

    elif model_type == "spline_poisson":
        model = SplinePoissonProcess(D, K, dt, dataset_stats=dataset_stats, **kw)

    elif model_type == "piecewise_constant":
        model = PiecewiseConstantPoissonProcess(D, K, dt, dataset_stats=dataset_stats, **kw)

    elif model_type == "hawkes":
        model = ConstBaselineExpKernelHawkesProcess(D=D, dataset_stats=dataset_stats, **kw)

    elif model_type == "linear_exp_hawkes":
        model = LinearBaselineExpKernelHawkesProcess(D=D, dataset_stats=dataset_stats, **kw)

    elif model_type == "spline_exp_hawkes":
        model = SplineBaselineExpKernelHawkesProcess(D=D, num_knots=K, delta_t=dt, dataset_stats=dataset_stats, **kw)

    elif model_type == "piecewise_exp_hawkes":
        model = PiecewiseBaselineExpKernelHawkesProcess(D=D, num_knots=K, delta_t=dt, dataset_stats=dataset_stats, **kw)

    elif model_type == "numerical_spline_exp_hawkes":
        model = NumericalSplineBaselineExpKernelHawkesProcess(
            D=D, num_knots=K, delta_t=dt, dataset_stats=dataset_stats, **kw
        )

    elif model_type == "inhibitive_const_exp_hawkes":
        model = InhibitiveConstExpHawkesProcess(
            D=D,
            clamping_fn=arch.inhibitive_clamping_fn,
            constrain_kernel=arch.constrain_kernel,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "inhibitive_spline_exp_hawkes":
        model = InhibitiveSplineExpHawkesProcess(
            D=D,
            num_knots=K,
            delta_t=dt,
            clamping_fn=arch.inhibitive_clamping_fn,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "const_spline_kernel":
        model = ConstBaselineSplineKernelHawkesProcess(
            D=D, num_knots=kK, delta_t=kdt, dataset_stats=dataset_stats, **kw
        )

    elif model_type == "const_piecewise_kernel":
        model = ConstBaselinePiecewiseKernelHawkesProcess(
            D=D, num_knots=kK, delta_t=kdt, dataset_stats=dataset_stats, **kw
        )

    elif model_type == "spline_spline_kernel":
        model = SplineBaselineSplineKernelHawkesProcess(
            D=D,
            num_knots_baseline=K,
            num_knots_kernel=kK,
            delta_t_baseline=dt,
            delta_t_kernel=kdt,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "spline_ns_spline_kernel":
        # Absolute-time (non-stationary) axis uses the baseline time anchor / uniform
        # spacing; the Δt axis reuses the kernel knots (kK, kdt).
        ns_dt = T_baseline / (arch.spline_ns_K - 1) if arch.spline_ns_K > 1 else T_baseline
        model = SplineBaselineNSSplineKernelHawkesProcess(
            D=D,
            num_knots_baseline=K,
            num_knots_ns=arch.spline_ns_K,
            num_knots_kernel=kK,
            delta_t_baseline=dt,
            delta_t_ns=ns_dt,
            delta_t_kernel=kdt,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "ns_exp_hawkes":
        # Non-stationary exp kernel: α(t), β(t) knots live on the absolute-time axis,
        # so they use BASELINE-scale spacing (T_baseline anchor), NOT the Δt-kernel (kK, kdt).
        # spline_ns_K sets the absolute-time knot count (shared with spline_ns_spline_kernel).
        ns_dt = T_baseline / (arch.spline_ns_K - 1) if arch.spline_ns_K > 1 else T_baseline
        model = ConstBaselineNSExpKernelHawkesProcess(
            D=D, num_knots=arch.spline_ns_K, delta_t=ns_dt, dataset_stats=dataset_stats, **kw
        )

    elif model_type == "spline_ns_exp_hawkes":
        # Spline baseline (spline_K knots) + NS-exp kernel whose α(t), β(t) absolute-time
        # knots use BASELINE-scale spacing and spline_ns_K knots (NOT the Δt-kernel kK/kdt).
        ns_dt = T_baseline / (arch.spline_ns_K - 1) if arch.spline_ns_K > 1 else T_baseline
        model = SplineBaselineNSExpKernelHawkesProcess(
            D=D,
            num_knots_baseline=K,
            num_knots_kernel=arch.spline_ns_K,
            delta_t_baseline=dt,
            delta_t_kernel=ns_dt,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "inhibitive_spline_spline_kernel":
        model = InhibitiveSplineBaselineSplineKernelHawkesProcess(
            D=D,
            num_knots_baseline=K,
            num_knots_kernel=kK,
            delta_t_baseline=dt,
            delta_t_kernel=kdt,
            clamping_fn=arch.inhibitive_clamping_fn,
            dataset_stats=dataset_stats,
            **kw,
        )

    elif model_type == "neural_hawkes":
        model = SplineBaselineNeuralKernelHawkesProcess(
            D=D,
            num_knots=K,
            delta_t=dt,
            clamping_fn=arch.inhibitive_clamping_fn,
            dataset_stats=dataset_stats,
            kernel_emb_dim=arch.neural_kernel_emb_dim,
            kernel_time_embd_dim=arch.neural_kernel_time_embd_dim,
            kernel_n_hidden=arch.neural_kernel_n_hidden,
            kernel_n_mlp_layers=arch.neural_kernel_n_mlp_layers,
            **kw,
        )

    elif model_type == "neural_spline_hawkes":
        model = SplineBaselineNeuralSplineKernelHawkesProcess(
            D=D,
            num_baseline_knots=K,
            baseline_delta_t=dt,
            num_kernel_knots=kK,
            kernel_delta_t=kdt,
            clamping_fn=arch.inhibitive_clamping_fn,
            dataset_stats=dataset_stats,
            kernel_emb_dim=arch.neural_kernel_emb_dim,
            kernel_n_hidden=arch.neural_kernel_n_hidden,
            kernel_n_mlp_layers=arch.neural_kernel_n_mlp_layers,
            **kw,
        )

    elif model_type in (
        "neural_poisson",
        "neural_log_linear",
        "neural_spline",
        "neural_thp",
        "neural_sahp",
        "neural_mlp",
        "neural_piecewise_constant",
    ):
        cfg = TransformerConfig(
            num_event_types=D,
            n_layer=arch.n_layer,
            n_head=arch.n_head,
            n_embd=arch.n_embd,
            dropout=arch.dropout,
        )
        if model_type == "neural_poisson":
            model = NeuralPoissonProcess(cfg, link=arch.neural_poisson_link, **kw)
        elif model_type == "neural_log_linear":
            model = NeuralLogLinearProcess(cfg, **kw)
        elif model_type == "neural_spline":
            model = NeuralSplineProcess(
                cfg,
                num_spline_bins=arch.spline_kernel_K,
                knot_gaps=neural_kdt,
                dataset_stats=dataset_stats,
                **kw,
            )
        elif model_type == "neural_thp":
            model = NeuralTHPProcess(cfg, **kw)
        elif model_type == "neural_sahp":
            model = NeuralSAHPProcess(cfg, **kw)
        elif model_type == "neural_mlp":
            model = NeuralMLPProcess(
                cfg,
                n_hidden=arch.mlp_n_hidden,
                n_mlp_layers=arch.mlp_n_layers,
                time_embd_dim=arch.mlp_time_embd_dim,
                **kw,
            )
        elif model_type == "neural_piecewise_constant":
            model = NeuralPiecewiseConstantProcess(
                cfg,
                num_bins=arch.spline_kernel_K,
                knot_gaps=neural_kdt,
                dataset_stats=dataset_stats,
                **kw,
            )

    elif model_type in ("easytpp_thp", "easytpp_sahp"):
        cfg = TransformerConfig(
            num_event_types=D,
            n_layer=arch.n_layer,
            n_head=arch.n_head,
            n_embd=arch.n_embd,
            dropout=arch.dropout,
        )
        if model_type == "easytpp_thp":
            model = EasyTPPTHPProcess(cfg, **kw)
        elif model_type == "easytpp_sahp":
            model = EasyTPPSAHPProcess(cfg, **kw)

    elif model_type == "intensity_free":
        if arch.iftpp_encoder == "transformer":
            encoder = TransformerEncoder(
                TransformerConfig(
                    num_event_types=D,
                    n_layer=arch.n_layer,
                    n_head=arch.n_head,
                    n_embd=arch.n_embd,
                    dropout=arch.dropout,
                )
            )
        elif arch.iftpp_encoder == "gru":
            encoder = GRUEncoder(
                num_event_types=D,
                n_embd=arch.n_embd,
                num_layers=arch.n_layer,
                dropout=arch.dropout,
            )
        else:
            raise ValueError(f"Unknown iftpp_encoder '{arch.iftpp_encoder}' (expected 'gru' or 'transformer').")
        # Inter-event standardisation: prefer data-driven stats (mean/std of log
        # inter-event gaps on the scaled gaps the model sees); fall back to the
        # arch defaults when unavailable or degenerate.  Mirrors the
        # spline_T_factor/dataset_stats fallback above.
        if arch.iftpp_standardize and dataset_stats is not None and dataset_stats.std_log_inter_time > 0:
            mean_log_inter_time = dataset_stats.mean_log_inter_time
            std_log_inter_time = dataset_stats.std_log_inter_time
        else:
            mean_log_inter_time = arch.mean_log_inter_time
            std_log_inter_time = arch.std_log_inter_time
        model = IntensityFreeProcess(
            encoder,
            num_mix_components=arch.num_mix_components,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            **kw,
        )

    elif model_type == "nhp":
        encoder = ContTimeLSTMEncoder(num_event_types=D, hidden_dim=arch.n_embd)
        model = NHPProcess(encoder, **kw)

    elif model_type == "attnhp":
        encoder = AttNHPEncoder(
            num_event_types=D,
            d_model=arch.n_embd,
            d_time=arch.attnhp_time_emb_size,
            n_layers=arch.n_layer,
            n_head=arch.n_head,
            dropout=arch.dropout,
            use_norm=arch.attnhp_use_ln,
        )
        model = AttNHPProcess(encoder, **kw)

    else:
        raise AssertionError(f"Unhandled model_type '{model_type}'")  # unreachable

    # CIConfig-D: the user-facing arch knobs (num_ci_integration_steps, AttNHP chunk) are
    # consumed here — inject them into numerical models' CIConfig after construction, leaving
    # each model's natural placement/rule intact.  Closed-form models keep their default;
    # density models (DensityBasedTPP, e.g. intensity_free) have no CIConfig — skip.
    if hasattr(model, "ci_config") and model.ci_config.mode == "numerical":
        if model_type == "attnhp":
            model.ci_config = replace(
                model.ci_config, n_points=ci, chunk_size=(arch.attnhp_quadrature_chunk_size or None)
            )
        else:
            model.ci_config = replace(model.ci_config, n_points=ci)

    if load_path is not None:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        print(f"Loading weights from {load_path}...")
        model.load_state_dict(torch.load(str(load_path), map_location="cpu", weights_only=True))

    return model, run_name
