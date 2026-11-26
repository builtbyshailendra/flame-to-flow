"""The known answers for the channel, and the helpers that compare against them."""
import numpy as np
import pytest

from f2f import channel, literature


def test_channel_numbers():
    ch = channel.Channel(nu=2.414e-4, rho=0.2245, re=200)
    assert ch.u_mean == pytest.approx(200 * 2.414e-4 / 0.01)
    assert ch.dpdx_kinematic == pytest.approx(-12 * ch.nu * ch.u_mean / ch.h ** 2)
    # Pa from kinematic: multiply by rho (mu = rho nu)
    assert ch.dpdx_pa == pytest.approx(-12 * (ch.nu * ch.rho) * ch.u_mean / ch.h ** 2)
    assert ch.length == pytest.approx(0.3)


def test_durst_entrance_length():
    h = 0.01
    assert literature.durst_entrance_length(200, h) / h == pytest.approx(8.92, abs=0.01)
    assert literature.durst_entrance_length(1e-6, h) / h == pytest.approx(0.631, abs=1e-3)
    # large Re: linear, 0.0442 Re h
    assert literature.durst_entrance_length(2000, h) / h == pytest.approx(88.4, rel=0.001)


def test_exact_parabola_scores_perfectly():
    h, U = 0.01, 3.0
    y = (np.arange(20) + 0.5) * h / 20
    m = channel.profile_metrics(y, channel.poiseuille(y, h, U), h, U)
    assert m["umax_ratio"] == pytest.approx(1.5, rel=1e-12)
    assert m["l2_rel"] < 1e-12


@pytest.mark.parametrize("ny", [10, 20, 40, 80])
def test_discrete_solution_errors_are_second_order(ny):
    # Solving the 1-D finite-volume equations gives dp/dx exactly G_exact / (1 + 2/ny^2):
    # the half-cell wall gradient is what makes the discrete answer differ.
    y, u, g = channel.discrete_poiseuille(ny)
    assert g / -12.0 == pytest.approx(1 / (1 + 2 / ny ** 2), rel=1e-10)
    assert u.mean() == pytest.approx(1.0, rel=1e-12)
    l2 = channel.profile_metrics(y, u, 1.0, 1.0)["l2_rel"]
    assert l2 == pytest.approx(0.93 / ny ** 2, rel=0.04)


def test_pressure_gradient_fit():
    x = np.linspace(0, 0.3, 301)
    p = 5.0 - 140.0 * x
    p[x < 0.05] += 3.0                    # entrance region, excluded by the fit window
    d = channel.pressure_gradient(x, p, 0.15, 0.27)
    assert d["dpdx"] == pytest.approx(-140.0, rel=1e-10)


def test_entrance_length_from_centreline():
    x = np.linspace(0, 0.3, 3001)
    uc = 1.5 * (1 - 0.5 * np.exp(-x / 0.02))   # reaches 99% where exp(-x/0.02) = 0.02
    d = channel.entrance_length(x, uc, 0.2, 0.27)
    assert d["L_e"] == pytest.approx(-0.02 * np.log(0.02), rel=1e-3)


def test_observed_order():
    assert channel.observed_order([4e-2, 1e-2, 2.5e-3]) == pytest.approx([2.0, 2.0])
