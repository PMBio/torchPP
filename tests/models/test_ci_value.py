"""Per-family NUMERICAL-VALUE correctness for the cumulative-intensity and
marginal machinery.

Unlike the universal relations in ``tests/contract/test_model_contract.py``,
these checks compare a model's numerical output against a ground truth whose
construction differs per family (finite-difference d/dt Λ ≈ λ, marginal
quadrature summing to 1, analytical-vs-empirical sampling marginals, inverse-CDF
boundary magnitudes). They are therefore kept fanned over the FULL model matrix
(``model`` / ``model_type``) and carry the family-specific skip predicates.
"""

import math

import pytest
import torch
from helpers import DEVICE, STEP_INTENSITY_MODEL_TYPES, HEAVY_TAIL_DENSITY_MODEL_TYPES, skip_if_density


class TestCumulativeIntensity:
    """Per-family numerical checks on cumulative intensity."""

    def test_cumulative_intensity_derivative_relation(
        self, model, model_type, test_data, tolerances, numerical_settings
    ):
        """Test that numerical derivative of CI approximates intensity."""
        # cumulative_intensity (and ci_config) are intensity-family concepts; a
        # density-based model (IFTPP) has neither, so the CI-derivative relation is
        # ill-posed for it — exclude it from this fan before touching ci_config.
        skip_if_density(model)
        # Skip for models using numerical CI to avoid compounding errors
        if not (model.ci_config.mode == "closed_form"):
            pytest.skip("Skipping numerical derivative test for numerical CI model")
        # Skip step-function-intensity models: their CI is piecewise-linear, so a
        # finite-difference derivative is wrong near the knot discontinuities.
        if model_type in STEP_INTENSITY_MODEL_TYPES:
            pytest.skip("Skipping numerical derivative test for step-function-intensity model")
        if model_type in HEAVY_TAIL_DENSITY_MODEL_TYPES:
            pytest.skip("Heavy-tailed density: finite-difference d/dt CI inaccurate near tau->0")

        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        # Start after last event to avoid discontinuities. Intensity can jump at obersved event times, which would make numerical derivative inaccurate. By starting after last event, we test in a smooth region.
        last_time = seq1.time_points.max().item()
        times = torch.linspace(
            last_time + numerical_settings["epsilon"],
            last_time + 2.0 + numerical_settings["epsilon"],
            steps=numerical_settings["integration_steps"],
        ).to(DEVICE)
        delta_t = times[1] - times[0]

        # Compute numerical derivative
        if hasattr(model, "prepare_batch"):
            model.prepare_batch(batch_single)
        cis = torch.stack([model.cumulative_intensity(t.unsqueeze(0), batch_single) for t in times])
        if hasattr(model, "clear_batch"):
            model.clear_batch()
        cis = cis.squeeze()
        ci_derivative = (cis[1:] - cis[:-1]) / delta_t

        # Compute actual intensity
        intensities = torch.stack([model.intensity(t.unsqueeze(0), batch_single) for t in times[:-1]])
        intensities = intensities.squeeze()

        assert torch.allclose(ci_derivative.float(), intensities.float(), atol=tolerances["standard"]), (
            "Numerical derivative of CI should approximate intensity"
        )


class TestInverseCDF:
    """Per-family numerical checks on inverse CDF."""

    def test_inverse_cdf_boundary_values(self, model, model_type, test_data, d):
        """Test inverse CDF at boundary probability values."""
        if model_type in HEAVY_TAIL_DENSITY_MODEL_TYPES:
            pytest.skip("Heavy-tailed inter-event density: CDF-inversion boundary accuracy is loose")
        batch_single = test_data["batch_single"]

        # At probability 0, should give time close to last event
        probs_zero = torch.zeros(1, device=DEVICE)
        times_zero = model.time_quantile(probs_zero, batch_single)  # (B,)
        assert torch.all(torch.isfinite(times_zero)), "Inverse CDF at p=0 should be finite"
        assert torch.allclose(times_zero.float(), batch_single.last_event_time.float(), atol=1e-3), (
            "Inverse CDF at p=0 should be close to last event time"
        )

        # At probability near 1, should give large times
        probs_high = torch.ones(1, device=DEVICE) * 0.99
        times_high = model.time_quantile(probs_high, batch_single)
        assert torch.all(torch.isfinite(times_high)), "Inverse CDF at p=0.99 should be finite"
        assert torch.all(times_high > times_zero), "Higher probability should give later times"


class TestMarginalDistributions:
    """Per-family numerical checks on marginal distributions."""

    def test_type_marginals_sum_to_one(self, model, model_type, test_data, tolerances, numerical_settings):
        """Test that type marginals integrate to approximately 1 over time."""
        if model_type in HEAVY_TAIL_DENSITY_MODEL_TYPES:
            pytest.skip("Heavy-tailed density: coarse-grid marginal quadrature undershoots (fine grid -> 1)")
        batch_single = test_data["batch_single"]

        # Find the 0.999 quantile time to get reasonable upper bound
        t_999 = model.time_quantile(torch.tensor([0.999]).to(DEVICE), batch_single).item()

        last_time = batch_single.last_event_time.item()
        times = torch.linspace(
            last_time + numerical_settings["epsilon"], t_999, steps=numerical_settings["integration_steps"]
        ).to(DEVICE)

        # Compute joint PDF over time and type
        with torch.no_grad():  # values only: retained graphs over the grid would cost GBs
            joint_time_type_pdf = torch.stack(
                [model.density_joint(t.unsqueeze(0), batch_single, log=False) for t in times]
            )
        joint_time_type_pdf = joint_time_type_pdf.squeeze()  # Shape: (num_times, D)

        # Integrate over time to get type marginals
        type_marginals = torch.trapezoid(joint_time_type_pdf, times, dim=0)  # Shape: (D,)

        # Type marginals should sum to approximately 1.0 (since we integrate to 0.999 quantile)
        # Upper bound slightly above 1.0 to account for trapezoidal rule overshoot
        # on piecewise-constant PDFs (kinks at knot boundaries)
        assert 0.98 <= type_marginals.sum() <= 1.01, (
            f"Type marginals should sum to ~1.0, got {type_marginals.sum().item():.4f}"
        )
        assert torch.all(type_marginals >= 0), "Type marginals should be non-negative"

        # Check if model implementation of marginal class distribution matches integrated type marginals
        model_class_dist = model._marginal_class_distribution(
            batch_single, num_points=numerical_settings["integration_steps"]
        )
        assert torch.allclose(type_marginals.float(), model_class_dist.float(), atol=tolerances["sampling"]), (
            f"Marginal class distribution {model_class_dist} should match integrated type marginals {type_marginals}"
        )

    def test_time_marginal_cdf_properties(self, model, test_data, tolerances, numerical_settings):
        """Test that time marginal CDF has proper properties."""
        batch_single = test_data["batch_single"]

        # Find the 0.999 quantile time
        t_999 = model.time_quantile(torch.tensor([0.999]).to(DEVICE), batch_single).item()

        # Log-spaced grid: dense right after the last event, where the density can peak sharply.
        last_time = batch_single.last_event_time.item()
        tau = torch.logspace(
            math.log10(numerical_settings["epsilon"]),
            math.log10(t_999 - last_time),
            steps=numerical_settings["integration_steps"],
            dtype=torch.float64,
        )
        times = (last_time + tau).to(DEVICE)

        # Compute joint PDF over time and type
        with torch.no_grad():  # values only: retained graphs over the grid would cost GBs
            joint_time_type_pdf = torch.stack(
                [model.density_joint(t.unsqueeze(0), batch_single, log=False) for t in times]
            )
        joint_time_type_pdf = joint_time_type_pdf.squeeze()  # Shape: (num_times, D)

        # Sum over event types to get time marginal (density)
        time_marginal = joint_time_type_pdf.sum(dim=1)  # Shape: (num_times,)

        # Integrate to get time CDF
        time_cdf = torch.cumulative_trapezoid(time_marginal, times, dim=0)

        assert torch.all(time_cdf >= 0), "Time CDF should be non-negative"
        assert torch.all(time_cdf <= 1 + 0.01), "Time CDF should be <= 1"
        assert 0.98 <= time_cdf[-1] <= 1.01, (
            f"Time CDF at 0.999 quantile should be ~0.999, got {time_cdf[-1].item():.4f}"
        )

        # Check monotonicity
        diffs = time_cdf[1:] - time_cdf[:-1]
        assert torch.all(diffs >= -1e-6), "Time CDF should be monotonic"

        with torch.no_grad():
            model_CDFs = torch.stack([model.cdf_time(t.unsqueeze(0), batch_single, log=False).sum() for t in times])
        assert torch.allclose(time_cdf.float(), model_CDFs[1:].float(), atol=tolerances["sampling"]), (
            "Time CDF from integration should match model CDF"
        )

    def test_marginals_vs_sampling(self, model, model_type, test_data, tolerances, numerical_settings):
        """Test that analytical marginals match empirical distributions from sampling."""

        if model_type in HEAVY_TAIL_DENSITY_MODEL_TYPES:
            pytest.skip("Heavy-tailed density: empirical-vs-analytical CDF match needs a fine grid")
        if not (model.ci_config.mode == "closed_form"):
            pytest.skip("Skipping marginal vs sampling test for numerical CI model due to compounded numerical errors")

        batch_single = test_data["batch_single"]
        seq1 = test_data["seq1"]

        # Find the 0.999 quantile time
        t_999 = model.time_quantile(torch.tensor([0.999]).to(DEVICE), batch_single).item()

        last_time = batch_single.last_event_time.item()
        times = torch.linspace(
            last_time + numerical_settings["epsilon"],
            t_999,
            steps=100,  # Use fewer points for comparison
        ).to(DEVICE)

        # Compute analytical marginals
        with torch.no_grad():  # values only: retained graphs over the grid would cost GBs
            joint_time_type_pdf = torch.stack(
                [model.density_joint(t.unsqueeze(0), batch_single, log=False) for t in times]
            )
        joint_time_type_pdf = joint_time_type_pdf.squeeze()  # Shape: (num_times, D)

        # Type marginals (normalized)
        type_marginals = torch.trapezoid(joint_time_type_pdf, times, dim=0)
        type_marginals = type_marginals / type_marginals.sum()  # Normalize to sum to 1

        # Time CDF
        time_marginal = joint_time_type_pdf.sum(dim=1)
        time_cdf = torch.cumulative_trapezoid(time_marginal, times, dim=0)

        # Sample continuations and compute empirical marginals
        num_samples = 1000
        tstar, dstar = [], []
        for i in range(num_samples):
            _, tt, dd = model.sample(seq1, num_steps=1, rng=i)
            tstar.append(tt[0])
            dstar.append(dd[0])

        tstar = torch.tensor(tstar).to(DEVICE)
        dstar = torch.stack(dstar)

        # Empirical type marginals
        empirical_type_marginals = torch.mean(dstar, dim=0)

        # Empirical time CDF
        empirical_time_cdf = torch.tensor([(tstar <= t).float().mean() for t in times]).to(DEVICE)

        # Compare type marginals
        assert torch.allclose(type_marginals.float(), empirical_type_marginals.float(), atol=tolerances["sampling"]), (
            f"Empirical type marginals {empirical_type_marginals} should match analytical {type_marginals}"
        )

        # Compare time CDFs (skip first point due to cumulative_trapezoid behavior)
        assert torch.allclose(time_cdf.float(), empirical_time_cdf[1:].float(), atol=tolerances["sampling"]), (
            "Empirical time CDF should match analytical time CDF"
        )
