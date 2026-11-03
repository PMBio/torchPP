"""Regression snapshot of the universal-contract outputs.

Every model wraps a scaled *core* in a native-unit *shell*. With ``time_scale`` at
its default ``1.0`` (what ``create_model`` yields without dataset stats) the shell
is a no-op. This test pins a compact ``(sum, sum-of-squares)`` fingerprint of each
contract method for the six archetypes on a fixed seed and batch, so any
unintended numerical change on the default path fails.

Regenerate the ``SNAPSHOT`` literal with ``tests/capture_number_preservation.py``
(only when a change is intended).
"""

import torch

from helpers import ARCHETYPES, build_test_data, create_model
from torchpp.tpp.base_tpp import IntensityBasedTPP

# Reference fingerprints (CPU, seed=42, time_scale=1.0).
SNAPSHOT = {
    'const_exp_hawkes': {
        'likelihood': (-111.5197166586413, 6218.4714476811005),
        'density_time': (-20.75729139626928, 225.0638578828578),
        'density_joint': (-69.2693672309953, 829.5128547492886),
        'survival_time': (-26.975325082032885, 373.431700149262),
        'cdf_time': (-1.2564870734643251e-05, 1.5402298647462213e-10),
        'prob_mark': (-6.982559079196192, 8.997204287230696),
        'prob_mark_given_time': (-6.99749304218747, 9.067550743585961),
        'time_mean': (4.888629163136363, 11.96926490684505),
        'time_quantile': (4.861334285346362, 11.836226435505374),
        'intensity': (44.798353718064696, 370.02698139086465),
        'cumulative_intensity': (133.5295014045696, 3288.2109022129343),
    },
    'inhibitive_const_exp_hawkes': {
        'likelihood': (-111.62766700658659, 6230.514733761298),
        'density_time': (-20.77632983641247, 225.47644207461008),
        'density_joint': (-69.32247234684837, 830.7811673847557),
        'survival_time': (-26.99595800653928, 374.0058393791285),
        'cdf_time': (-1.2459935215815697e-05, 1.514756796008864e-10),
        'prob_mark': (-6.979019780293742, 8.980560530732546),
        'prob_mark_given_time': (-6.993482837610975, 9.04863462159852),
        'time_mean': (4.888565148505588, 11.968952473375886),
        'time_quantile': (4.861290164409562, 11.83601231910484),
        'intensity': (44.83407996130437, 370.31964536304537),
        'cumulative_intensity': (133.64286362196702, 3291.019893818516),
    },
    'piecewise_poisson': {
        'likelihood': (-15.785835139999064, 125.81620619450675),
        'density_time': (-1.0395692720151395, 0.6018800726519321),
        'density_joint': (-9.825389579965252, 16.495484366521204),
        'survival_time': (-2.000470206483728, 2.0624684605074126),
        'cdf_time': (-0.9458690615435081, 0.4689092390257178),
        'prob_mark': (-6.8350084019813, 8.314397720669865),
        'prob_mark_given_time': (-6.706681763919834, 7.717783869610054),
        'time_mean': (5.965425707879724, 17.816351951543353),
        'time_quantile': (5.680267612061886, 16.146174966659423),
        'intensity': (3.233605146408081, 1.814477383467267),
        'cumulative_intensity': (12.024979989776284, 24.487456148229658),
    },
    'nhp': {
        'likelihood': (-13.68138820343281, 93.96573781076769),
        'density_time': (-0.984546840335677, 0.5531856404521395),
        'density_joint': (-9.628806895661594, 15.82395339016071),
        'survival_time': (-2.218315603550355, 2.5215328587463004),
        'cdf_time': (-0.8232066834481389, 0.354084204044022),
        'prob_mark': (-6.675982820158039, 7.59260362327138),
        'prob_mark_given_time': (-6.675166374654564, 7.589242711004244),
        'time_mean': (5.878532461377203, 17.296427494789427),
        'time_quantile': (5.549009634026973, 15.414411101638585),
        'intensity': (3.7064892053604126, 2.3565430882380767),
        'cumulative_intensity': (11.182683287194026, 21.409296517919),
    },
    'intensity_free': {
        'likelihood': (-14.246599979733524, 101.72204642156785),
        'density_time': (-1.6197555096775982, 1.3958451427862197),
        'density_joint': (-11.485092748040964, 22.316578654416464),
        'survival_time': (-0.7401720829352652, 0.2855590357737978),
        'cdf_time': (-2.3896341264748457, 2.9152230591032557),
        'prob_mark': (-6.625826001167297, 7.383402143558115),
        'prob_mark_given_time': (-6.62582621900817, 7.383402626400123),
        'time_mean': (10.275218594808637, 53.425321659005064),
        'time_quantile': (7.051914660825566, 24.870615705715878),
    },
    'neural_log_linear': {
        'likelihood': (-16.183766326370307, 130.95747011586292),
        'density_time': (-1.021278928064512, 0.6114675656498738),
        'density_joint': (-10.49546303928738, 20.251260811912097),
        'survival_time': (-3.1393913910814293, 4.98033195003818),
        'cdf_time': (-0.4754209713331646, 0.11669607997535916),
        'prob_mark': (-7.431626255093843, 10.947405126974665),
        'prob_mark_given_time': (-7.4316262550938434, 10.947405126974665),
        'time_mean': (5.547539114739031, 15.398225046849312),
        'time_quantile': (5.353620410828086, 14.343028257299984),
        'intensity': (5.7745524903132015, 7.08919701837481),
        'cumulative_intensity': (21.10671208805091, 99.3397839972097),
    },
}


def _fp(x: torch.Tensor):
    """(sum, sum-of-squares) in float64 — a compact bit-identity fingerprint."""
    xd = x.detach().double()
    return (float(xd.sum().item()), float((xd * xd).sum().item()))


def _fingerprints_for(name: str) -> dict:
    """Contract-method fingerprints for one archetype, forced onto CPU (device-stable)."""
    data = build_test_data(torch.device("cpu"))
    ts = data["batch_multi"]
    B = ts.time_points.shape[0]
    t = torch.full((B,), 3.0, dtype=torch.float64)
    T = torch.full((B,), 3.0, dtype=torch.float64)

    model = create_model(name).to("cpu")
    model.eval()
    out = {}
    with torch.no_grad():
        out["likelihood"] = _fp(model.likelihood(ts, T=T, log=True))
        out["density_time"] = _fp(model.density_time(t, ts, log=True))
        out["density_joint"] = _fp(model.density_joint(t, ts, log=True))
        out["survival_time"] = _fp(model.survival_time(t, ts, log=True))
        out["cdf_time"] = _fp(model.cdf_time(t, ts, log=True))
        out["prob_mark"] = _fp(model.prob_mark(ts, log=True))
        out["prob_mark_given_time"] = _fp(model.prob_mark_given_time(t, ts, log=True))
        out["time_mean"] = _fp(model.time_mean(ts))
        out["time_quantile"] = _fp(model.time_quantile(0.5, ts))
        if isinstance(model, IntensityBasedTPP):
            out["intensity"] = _fp(model.intensity(t, ts))
            out["cumulative_intensity"] = _fp(model.cumulative_intensity(T, ts))
    return out


import pytest  # noqa: E402


@pytest.mark.parametrize("name", ARCHETYPES)
def test_contract_outputs_bit_identical_at_unit_scale(name):
    """Every contract fingerprint matches the reference at time_scale=1.0."""
    got = _fingerprints_for(name)
    ref = SNAPSHOT[name]
    assert set(got) == set(ref), f"{name}: method set drifted: {set(got) ^ set(ref)}"
    for method, (s, sq) in got.items():
        rs, rsq = ref[method]
        assert abs(s - rs) <= 1e-6 + 1e-6 * abs(rs), f"{name}.{method} sum: {s} vs {rs}"
        assert abs(sq - rsq) <= 1e-6 + 1e-6 * abs(rsq), f"{name}.{method} sumsq: {sq} vs {rsq}"
