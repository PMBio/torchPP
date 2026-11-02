"""Unit tests for build_intensity_grid() / IntensityGrid.

IntensityGrid is a utility whose behaviour does not vary per model family, so it
fans over the ARCHETYPE set rather than the full matrix.  The per-property skips
(step-function intensity, non-analytical CI) are gated on the archetype name.
"""

import pytest
import torch
from helpers import DEVICE, STEP_INTENSITY_MODEL_TYPES, skip_if_density


class TestIntensityGrid:
    """Tests for build_intensity_grid() method."""

    @pytest.fixture(autouse=True)
    def _require_intensity_family(self, representative_model):
        # IntensityGrid is built from intensity/CI; density models expose neither.
        skip_if_density(representative_model)

    def test_build_intensity_grid_shape(self, representative_model, test_data, d):
        """Test grid has correct shape."""
        model = representative_model
        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        num_points = 50

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        assert grid.t_grid.shape == (1, num_points), f"t_grid shape mismatch: {grid.t_grid.shape}"
        assert grid.intensities.shape == (1, num_points, d), f"intensities shape mismatch: {grid.intensities.shape}"
        assert grid.ci_prefix.shape == (1, num_points, d), f"ci_prefix shape mismatch: {grid.ci_prefix.shape}"

    def test_build_intensity_grid_accuracy(self, representative_model, test_data, tolerances):
        """Test grid intensities match direct intensity calls."""
        model = representative_model
        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        num_points = 50

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        # Check a few random points
        for i in [0, num_points // 2, num_points - 1]:
            t_i = grid.t_grid[0, i]
            intensity_grid = grid.intensities[0, i, :]
            intensity_direct = model.intensity(t_i.unsqueeze(0), batch_single)[0]
            assert torch.allclose(intensity_grid.float(), intensity_direct.float(), atol=tolerances["tight"]), (
                f"Grid intensity at point {i} should match direct intensity()"
            )

    def test_build_intensity_grid_ci_consistency(self, representative_model, archetype, test_data, tolerances):
        """Test grid CI matches direct CI calls."""
        model = representative_model
        if not (model.ci_config.mode == "closed_form"):
            pytest.skip("Numerical CI models use grid internally - would be circular test")
        # Trapezoidal integration of a step-function intensity has O(h) error at
        # each discontinuity, so the grid CI won't match the exact analytical CI.
        if archetype in STEP_INTENSITY_MODEL_TYPES:
            pytest.skip("Grid CI vs analytical CI diverges for step-function-intensity models")

        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        # build_intensity_grid documents ≥200 points for negligible trapezoidal error;
        # at 100 the grid-vs-exact gap sits on the 0.01 tolerance edge and flips with
        # model-init RNG, so use the documented resolution.
        num_points = 200

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        # Check CI at end point
        ci_grid = grid.ci_prefix[0, -1, :]
        # t_start and t_end are already shape (1,) from batch_single.max_time
        ci_direct = model.cumulative_intensity(t_end, batch_single, T_low=t_start)[0]

        # Trapezoidal error is O(h²) and scales with the CI magnitude, which varies with
        # the (RNG-seeded) intensity steepness — so use a relative tolerance alongside the
        # absolute one; otherwise a steep instance tips a fixed atol on the tolerance edge.
        assert torch.allclose(ci_grid.float(), ci_direct.float(), rtol=1e-2, atol=tolerances["standard"]), (
            "Grid CI at end should match direct cumulative_intensity()"
        )

    def test_build_intensity_grid_monotonic_ci(self, representative_model, test_data):
        """Test that CI prefix is monotonically increasing."""
        model = representative_model
        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        num_points = 100

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        # Check monotonicity for each dimension
        for d_idx in range(grid.ci_prefix.shape[2]):
            ci_sequence = grid.ci_prefix[0, :, d_idx]
            diffs = ci_sequence[1:] - ci_sequence[:-1]
            assert torch.all(diffs >= -1e-8), f"CI should be monotonic for dimension {d_idx}"

    def test_build_intensity_grid_interpolation_intensity(self, representative_model, archetype, test_data, tolerances):
        """Test grid interpolation for intensity."""
        model = representative_model
        # Linear interpolation between grid points cannot reproduce a step-function
        # intensity near its discontinuities.
        if archetype in STEP_INTENSITY_MODEL_TYPES:
            pytest.skip("Linear grid interpolation can't reproduce a step-function intensity")

        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        num_points = 100

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        # Test intensity_at_t for a point in the middle
        t_mid = (t_start + t_end) / 2
        intensity_interp = grid.intensity_at_t(t_mid)[0]
        intensity_direct = model.intensity(t_mid, batch_single)[0]

        # Linear interpolation should be reasonably close
        # Use looser tolerance since we're interpolating
        assert torch.allclose(
            intensity_interp.float(), intensity_direct.float(), rtol=0.1, atol=tolerances["sampling"]
        ), "Interpolated intensity should be close to direct intensity"

    def test_build_intensity_grid_interpolation_ci(self, representative_model, test_data, tolerances):
        """Test grid interpolation for CI."""
        model = representative_model
        if not (model.ci_config.mode == "closed_form"):
            pytest.skip("Numerical CI models use grid internally")

        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0
        num_points = 100

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=num_points)

        # Test ci_at_t for a point in the middle
        t_mid = (t_start + t_end) / 2
        ci_interp = grid.ci_at_t(t_mid)[0]
        ci_direct = model.cumulative_intensity(t_mid, batch_single, T_low=t_start)[0]

        # Linear interpolation of CI should be close
        # Use looser tolerance since we're interpolating
        assert torch.allclose(ci_interp.float(), ci_direct.float(), rtol=0.1, atol=tolerances["sampling"]), (
            "Interpolated CI should be close to direct CI"
        )

    def test_build_intensity_grid_single_point(self, representative_model, test_data):
        """Test grid with single point."""
        model = representative_model
        batch_single = test_data["batch_single"]
        t_start = batch_single.max_time
        t_end = t_start + 2.0

        grid = model.build_intensity_grid(t_start, t_end, batch_single, num_points=1)

        # Single point should have CI = 0 (no interval to integrate over)
        assert torch.allclose(grid.ci_prefix, torch.zeros_like(grid.ci_prefix), atol=1e-10), (
            "Single point grid should have CI = 0"
        )

    def test_build_intensity_grid_batch_consistency(self, representative_model, test_data, tolerances):
        """Test batched grid building."""
        model = representative_model
        batch_multi = test_data["batch_multi"]
        seq1 = test_data["seq1"]
        seq2 = test_data["seq2"]

        t_start = batch_multi.max_time
        t_end = t_start + 2.0
        num_points = 50

        # Batched grid
        grid_batched = model.build_intensity_grid(t_start, t_end, batch_multi, num_points=num_points)

        # Individual grids
        from torchpp.utils.event_data import BatchedMVEventData

        batch1 = BatchedMVEventData([seq1.time_points], [seq1.event_types]).to(DEVICE)
        batch2 = BatchedMVEventData([seq2.time_points], [seq2.event_types]).to(DEVICE)

        grid1 = model.build_intensity_grid(t_start[0:1], t_end[0:1], batch1, num_points=num_points)
        grid2 = model.build_intensity_grid(t_start[1:2], t_end[1:2], batch2, num_points=num_points)

        # Check intensities match
        assert torch.allclose(
            grid_batched.intensities[0].float(), grid1.intensities[0].float(), atol=tolerances["tight"]
        )
        assert torch.allclose(
            grid_batched.intensities[1].float(), grid2.intensities[0].float(), atol=tolerances["tight"]
        )


def test_build_until_reaches_target_for_low_rate():
    """build_until must not silently truncate a low-rate process at the fixed chunk cap (audit N2).

    A constant rate λ=0.1 needs ~69 scaled units to reach CDF 0.999, far past the old
    ``chunk_duration=0.1 × max_chunks=100 = 10`` cap; the geometric continuation must extend the grid
    so the CDF target is met and ``E[t_next] = t_last + 1/λ`` is recovered.  Pre-fix this truncated
    to CDF≈0.63 / mean≈t_last+6.3 (~37% low).
    """
    from torchpp.utils.math import IntensityGrid
    from torchpp.utils.event_data import BatchedMVEventData
    from torchpp.tpp.base_tpp import IntensityBasedTPP

    rate = 0.1
    ts = BatchedMVEventData(
        [torch.tensor([0.5], dtype=torch.float64)], [torch.tensor([0], dtype=torch.long)]
    ).to(DEVICE)
    t_start = ts.max_time  # (1,)

    def intensity_func(t, ts):
        return torch.full((*t.shape, 1), rate, dtype=torch.float64, device=t.device)

    grid = IntensityGrid.build_until(
        intensity_func=intensity_func, t_start=t_start, ts=ts,
        points_per_unit_time=200.0, target_cdf=0.999, T_max=None,
    )
    total_ci = grid.ci_prefix[:, -1, :].sum(dim=1)  # (1,)
    cdf = 1.0 - torch.exp(-total_ci)
    assert float(cdf) >= 0.999, f"build_until truncated a low-rate process: CDF only {float(cdf):.4f}"
    mean = IntensityBasedTPP._mean_next_event_time_from_grid(grid)  # t_last + ∫S dτ ≈ t_last + 1/λ
    assert abs(float(mean) - (float(t_start) + 1.0 / rate)) < 0.5, (
        f"grid mean {float(mean):.3f} != analytic {float(t_start) + 1.0 / rate:.3f}"
    )
