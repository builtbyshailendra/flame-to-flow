"""Plane Poiseuille flow: the known answers for the OpenFOAM channel.

Fully developed laminar flow between plates a distance h apart, mean velocity U:
  u(y) = 6 U (y/h) (1 - y/h)          so u_max / U = 1.5
  dp/dx = -12 mu U / h^2              (OpenFOAM's incompressible p is p/rho: dp/dx = -12 nu U / h^2)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .literature import durst_entrance_length


@dataclass(frozen=True)
class Channel:
    """Geometry, fluid and flow of the 2-D channel case. Lengths in m, nu in m^2/s."""
    h: float = 0.01          # height
    length_h: float = 30.0   # length in channel heights
    dz_h: float = 0.1        # one-cell thickness in channel heights (any value works for 2-D)
    nu: float = 2.414e-4     # kinematic viscosity of the burnt gas at 1500 K (from Cantera)
    rho: float = 0.2245      # density of the burnt gas at 1500 K (from Cantera), for Pa
    re: float = 200.0        # Re = U h / nu, U the mean velocity

    @property
    def length(self) -> float:
        return self.length_h * self.h

    @property
    def dz(self) -> float:
        return self.dz_h * self.h

    @property
    def u_mean(self) -> float:
        return self.re * self.nu / self.h

    @property
    def dpdx_kinematic(self) -> float:
        """Fully developed pressure gradient of p/rho, m/s^2 (negative: pressure falls)."""
        return -12.0 * self.nu * self.u_mean / self.h ** 2

    @property
    def dpdx_pa(self) -> float:
        return self.rho * self.dpdx_kinematic

    @property
    def entrance_length(self) -> float:
        return durst_entrance_length(self.re, self.h)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(length=self.length, dz=self.dz, u_mean=self.u_mean,
                 dpdx_kinematic=self.dpdx_kinematic, dpdx_pa=self.dpdx_pa,
                 entrance_length=self.entrance_length)
        return d


@dataclass(frozen=True)
class Mesh:
    """Cells along (nx) and across (ny) the channel; one cell thick."""
    nx: int
    ny: int

    @property
    def cells(self) -> int:
        return self.nx * self.ny


# Three levels, both directions refined by 2 each time; dx/dy = 2.5 on every level.
LEVELS = {"coarse": Mesh(120, 10), "medium": Mesh(240, 20), "fine": Mesh(480, 40)}


def poiseuille(y: np.ndarray, h: float, u_mean: float) -> np.ndarray:
    eta = np.asarray(y) / h
    return 6.0 * u_mean * eta * (1.0 - eta)


def discrete_poiseuille(ny: int, h: float = 1.0, u_mean: float = 1.0, nu: float = 1.0):
    """What a second-order finite-volume code should give on ny uniform cells across the
    channel in the fully developed region: the 1-D discrete equations OpenFOAM solves there
    (central differences between cell centres; at a wall the gradient is taken over the
    half cell, (u_P - 0) / (dy/2)), with the mean velocity fixed.
    Returns (y_cell_centres, u_cells, dpdx_kinematic)."""
    dy = h / ny
    a = np.zeros((ny + 1, ny + 1))
    b = np.zeros(ny + 1)
    for j in range(ny):
        for nb, dist in ((j - 1, dy), (j + 1, dy)):
            if 0 <= nb < ny:
                a[j, j] -= nu / dist / dy
                a[j, nb] += nu / dist / dy
            else:                                  # wall face at half a cell
                a[j, j] -= nu / (dy / 2) / dy
        a[j, ny] = -1.0                            # nu u'' - dp/dx = 0
    a[ny, :ny] = 1.0 / ny                          # mean velocity
    b[ny] = u_mean
    sol = np.linalg.solve(a, b)
    y = (np.arange(ny) + 0.5) * dy
    return y, sol[:ny], sol[ny]


def profile_metrics(y: np.ndarray, u: np.ndarray, h: float, u_mean: float) -> dict:
    """Checks of a sampled cell-centre profile against Poiseuille.

    u_mean_sampled  mean of the cell values (uniform cells): should equal the inlet U (mass)
    umax_ratio      peak of a parabola fitted to the samples, over the inlet U (exact: 1.5)
    l2_rel          relative L2 error against 6U(y/h)(1-y/h) at the same points
    """
    y = np.asarray(y, float)
    u = np.asarray(u, float)
    exact = poiseuille(y, h, u_mean)
    c = np.polyfit(y, u, 2)
    u_peak = np.polyval(c, -c[1] / (2 * c[0]))
    return {
        "points": len(y),
        "u_mean_sampled": float(u.mean()),
        "umax_ratio": float(u_peak / u_mean),
        "umax_cell_ratio": float(u.max() / u_mean),
        "l2_rel": float(np.sqrt(np.mean((u - exact) ** 2) / np.mean(exact ** 2))),
        "max_abs_rel": float(np.max(np.abs(u - exact)) / (1.5 * u_mean)),
    }


def pressure_gradient(x: np.ndarray, p: np.ndarray, x_from: float, x_to: float) -> dict:
    """Least-squares slope of p(x) over [x_from, x_to] (the fully developed part)."""
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    m = (x >= x_from) & (x <= x_to)
    slope, intercept = np.polyfit(x[m], p[m], 1)
    resid = p[m] - (slope * x[m] + intercept)
    return {"dpdx": float(slope), "points": int(m.sum()),
            "max_resid_rel": float(np.max(np.abs(resid)) / (abs(slope) * (x_to - x_from)))}


def entrance_length(x: np.ndarray, uc: np.ndarray, x_fd_from: float, x_fd_to: float,
                    fraction: float = 0.99) -> dict:
    """Distance where the centreline velocity first reaches `fraction` of its fully developed
    value (the mean over [x_fd_from, x_fd_to]), by linear interpolation between samples."""
    x = np.asarray(x, float)
    uc = np.asarray(uc, float)
    m = (x >= x_fd_from) & (x <= x_fd_to)
    u_fd = float(uc[m].mean())
    target = fraction * u_fd
    above = np.nonzero(uc >= target)[0]
    if len(above) == 0 or above[0] == 0:
        return {"L_e": float("nan"), "u_fd": u_fd}
    i = above[0]
    x_e = x[i - 1] + (target - uc[i - 1]) * (x[i] - x[i - 1]) / (uc[i] - uc[i - 1])
    return {"L_e": float(x_e), "u_fd": u_fd}


def observed_order(errors: list[float], ratio: float = 2.0) -> list[float]:
    """Order of convergence between successive levels refined by `ratio`."""
    return [float(np.log(abs(a / b)) / np.log(ratio)) for a, b in zip(errors, errors[1:])]
