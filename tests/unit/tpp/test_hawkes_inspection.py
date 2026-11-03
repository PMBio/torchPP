"""Native-unit readouts of fitted baselines and kernels: shapes, layout, integrals, units."""

import pytest
import torch

import torchpp as tpp
from torchpp.tpp.hawkes import InhibitiveHawkesProcess

D = 3
CLOSED_FORM = [
    "const_exp_hawkes",
    "const_spline_hawkes",
    "const_piecewise_hawkes",
    "spline_spline_hawkes",
    "inhibitive_const_exp_hawkes",
    "inhibitive_spline_spline_hawkes",
    "inhibitive_spline_mlpspline_hawkes",
]


def _model(model_type, time_scale=1.0):
    m = tpp.create_model(model_type, D=D, seed=0)
    m.time_scale.fill_(time_scale)
    return m.double()


def _lags():
    return torch.cat([torch.zeros(1, dtype=torch.float64), torch.logspace(-6, 4, 100001, dtype=torch.float64)])


@pytest.mark.parametrize("model_type", CLOSED_FORM + ["inhibitive_spline_mlp_hawkes"])
def test_shapes(model_type):
    m = _model(model_type)
    with torch.no_grad():
        assert m.kernel_values(torch.tensor([0.0, 0.5, 2.0])).shape == (3, D, D)
        assert m.baseline_values(torch.tensor([0.0, 1.0])).shape == (2, D)


def test_exp_kernel_layout_is_target_source():
    m = _model("const_exp_hawkes")
    alpha, beta = m.kernel.transform_params()
    with torch.no_grad():
        k0 = m.kernel_values(torch.tensor([0.0], dtype=torch.float64))[0]
    torch.testing.assert_close(k0, alpha * beta)
    torch.testing.assert_close(m.branching_matrix(), alpha)


@pytest.mark.parametrize("model_type", CLOSED_FORM)
def test_branching_matrix_is_kernel_integral(model_type):
    m = _model(model_type)
    dt = _lags()
    with torch.no_grad():
        numeric = torch.trapezoid(m.kernel_values(dt), dt, dim=0)
        torch.testing.assert_close(m.branching_matrix(), numeric, rtol=1e-4, atol=2e-5)  # trapezoid error at piecewise jumps


def test_kernel_without_closed_form_needs_t_max():
    m = _model("inhibitive_spline_mlp_hawkes")
    with pytest.raises(ValueError, match="t_max"):
        m.branching_matrix()
    with torch.no_grad():
        assert m.branching_matrix(t_max=5.0).shape == (D, D)


@pytest.mark.parametrize("model_type", CLOSED_FORM + ["poisson", "spline_poisson", "loglinear_poisson"])
def test_native_units(model_type):
    """Rescaling time by c: rates scale by 1/c, pre-link terms do not, integrals by c only for pre-link terms."""
    c = 2.5
    m1, mc = _model(model_type), _model(model_type, time_scale=c)
    mc.load_state_dict(m1.state_dict())
    mc.time_scale.fill_(c)
    rate = not isinstance(m1, InhibitiveHawkesProcess)
    f = c if rate else 1.0
    t = torch.tensor([0.0, 0.3, 1.7], dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(mc.baseline_values(t * c) * f, m1.baseline_values(t))
        if hasattr(m1, "kernel_values"):
            torch.testing.assert_close(mc.kernel_values(t * c) * f, m1.kernel_values(t))
            torch.testing.assert_close(mc.branching_matrix() / (1.0 if rate else c), m1.branching_matrix())


def test_poisson_baseline_is_rate():
    m = _model("poisson", time_scale=4.0)
    (mu,) = m.baseline.transform_params()
    with torch.no_grad():
        torch.testing.assert_close(m.baseline_values(torch.tensor([1.0, 9.0]))[0], mu / 4.0)
