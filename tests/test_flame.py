"""Cantera against known answers: GRI-Mech 3.0 methane-air at 1 atm."""
import cantera as ct
import numpy as np
import pytest

from f2f import flame, literature


def test_stoichiometric_adiabatic_flame_temperature_is_about_2225_K():
    assert flame.adiabatic_flame(1.0)["T_ad"] == pytest.approx(2225, abs=10)


def test_flame_temperature_peaks_slightly_rich():
    rows = flame.flame_temperature_sweep(np.round(np.arange(0.6, 1.401, 0.05), 2))
    peak = max(rows, key=lambda r: r["T_ad"])
    assert 1.0 < peak["phi"] <= 1.1
    # lean and rich both cooler; dissociation keeps the peak well under the no-dissociation value
    assert rows[0]["T_ad"] < 1700 and rows[-1]["T_ad"] < 2000
    assert peak["T_ad"] < 2250


def test_lean_products_keep_oxygen_rich_products_keep_co():
    lean, rich = flame.adiabatic_flame(0.7), flame.adiabatic_flame(1.3)
    assert lean["X_O2"] > 0.05 and lean["X_CO"] < 1e-3
    assert rich["X_CO"] > 0.03 and rich["X_O2"] < 1e-3


@pytest.fixture(scope="module")
def speeds():
    # CI-size: the grid criteria from Cantera's own example, four mixtures
    rows = flame.flame_speed_sweep([0.8, 1.0, 1.1, 1.2], refine=flame.REFINE_EXAMPLE)
    return {r["phi"]: r for r in rows}


@pytest.mark.slow
def test_stoichiometric_flame_speed_is_036_to_039(speeds):
    assert 0.36 <= speeds[1.0]["S_L"] <= 0.39


@pytest.mark.slow
def test_flame_speed_peaks_slightly_rich(speeds):
    assert speeds[1.1]["S_L"] > speeds[1.0]["S_L"] * 0.99
    assert speeds[1.0]["S_L"] > speeds[0.8]["S_L"]
    assert speeds[1.1]["S_L"] > speeds[1.2]["S_L"]


@pytest.mark.slow
def test_flame_speed_within_8_percent_of_heat_flux_measurements(speeds):
    # measured at 298 K, computed at 300 K (about +1%); GRI-Mech 3.0 runs a few % fast here
    for phi, r in speeds.items():
        assert r["S_L"] == pytest.approx(literature.hermanns_speed(phi), rel=0.08), phi


@pytest.mark.slow
def test_flame_is_resolved(speeds):
    r = speeds[1.0]
    assert r["points"] > 100
    assert 0.3 < r["thickness_mm"] < 0.6          # thermal thickness of a methane-air flame
    assert r["T_end"] == pytest.approx(flame.adiabatic_flame(1.0)["T_ad"], abs=15)


def test_ignition_delay_is_arrhenius_like():
    rows = [flame.ignition_delay(T0) for T0 in (1200, 1400, 1600, 1800)]
    taus = [r["tau_s"] for r in rows]
    assert all(a > b for a, b in zip(taus, taus[1:]))
    assert taus[0] / taus[-1] > 100                 # two orders of magnitude over 600 K
    fit = flame.arrhenius_fit(rows)
    assert fit["r2"] > 0.995
    assert 150 < fit["E_kJ_per_mol"] < 220


def test_higher_pressure_ignites_sooner():
    assert flame.ignition_delay(1400, P=10 * ct.one_atm)["tau_s"] < \
        flame.ignition_delay(1400)["tau_s"] / 4


def test_product_properties_are_consistent():
    p = flame.product_properties(1.0, T_duct=1500)
    for k in ("flame", "duct", "duct_frozen"):
        d = p[k]
        # ideal gas: rho = P W / (R T)
        assert d["rho"] == pytest.approx(ct.one_atm * d["W"] / (ct.gas_constant * d["T"]), rel=1e-9)
        assert d["nu"] == pytest.approx(d["mu"] / d["rho"], rel=1e-12)
    assert p["flame"]["T"] == pytest.approx(2225, abs=10)
    assert p["duct"]["nu"] == pytest.approx(2.414e-4, rel=0.01)
    # gas viscosity rises with temperature, density falls
    assert p["flame"]["mu"] > p["duct"]["mu"] and p["flame"]["rho"] < p["duct"]["rho"]
    # recombination on cooling changes nu by under 1%
    assert p["duct"]["nu"] == pytest.approx(p["duct_frozen"]["nu"], rel=0.01)
