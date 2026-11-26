"""Methane-air combustion with Cantera and GRI-Mech 3.0.

Four calculations, each with a known answer to check against:
  adiabatic flame temperature  equilibrate at constant enthalpy and pressure ('HP')
  laminar flame speed          1-D freely propagating flame (FreeFlame), grid-refined
  ignition delay               constant-pressure reactor, time of the steepest temperature rise
  product properties           density and viscosity of the burnt gas, the OpenFOAM inputs
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cantera as ct
import numpy as np

MECH = "gri30.yaml"
FUEL = "CH4"
AIR = "O2:1.0, N2:3.76"
P_ATM = ct.one_atm


def gas(phi: float, T: float = 300.0, P: float = P_ATM) -> ct.Solution:
    """A fresh GRI-Mech 3.0 gas holding the unburnt methane-air mixture."""
    g = ct.Solution(MECH)
    g.TP = T, P
    g.set_equivalence_ratio(phi, FUEL, AIR)
    return g


# --- (a) adiabatic flame temperature -------------------------------------------------------

def adiabatic_flame(phi: float, T0: float = 300.0, P: float = P_ATM) -> dict:
    """Equilibrium products at constant enthalpy and pressure."""
    g = gas(phi, T0, P)
    g.equilibrate("HP")
    x = dict(zip(g.species_names, g.X))
    return {
        "phi": phi,
        "T_ad": g.T,
        "X_CO2": x["CO2"], "X_H2O": x["H2O"], "X_CO": x["CO"], "X_O2": x["O2"],
        "X_H2": x["H2"], "X_OH": x["OH"], "X_NO": x["NO"],
    }


def flame_temperature_sweep(phis, T0: float = 300.0, P: float = P_ATM) -> list[dict]:
    return [adiabatic_flame(float(p), T0, P) for p in phis]


# --- (b) laminar flame speed --------------------------------------------------------------

@dataclass(frozen=True)
class Refine:
    """Cantera's grid refinement criteria for a 1-D flame.

    ratio: largest allowed size ratio of neighbouring grid intervals
    slope: largest allowed change of any solution component between points, as a fraction of its range
    curve: the same for the change of slope
    """
    ratio: float
    slope: float
    curve: float

    def label(self) -> str:
        return f"ratio={self.ratio:g} slope={self.slope:g} curve={self.curve:g}"


# The settings Cantera's own adiabatic flame example uses, and two finer ones.
REFINE_EXAMPLE = Refine(3, 0.06, 0.12)
REFINE_SWEEP = Refine(2, 0.02, 0.04)      # used for the published sweep (results/flame_speed.csv)
GRID_STUDY = [
    Refine(3, 0.10, 0.20),
    Refine(3, 0.06, 0.12),
    Refine(3, 0.03, 0.06),
    Refine(2, 0.02, 0.04),
    Refine(2, 0.01, 0.02),
    Refine(2, 0.005, 0.01),
]


def _new_flame(phi: float, T0: float, P: float, width: float, transport: str) -> ct.FreeFlame:
    f = ct.FreeFlame(gas(phi, T0, P), width=width)
    f.transport_model = transport
    f.set_max_grid_points(f.flame, 5000)
    return f


def _flame_row(f: ct.FreeFlame, phi: float, refine: Refine, seconds: float) -> dict:
    T = f.T
    x = f.grid
    dTdx = np.gradient(T, x)
    return {
        "phi": phi,
        "S_L": float(f.velocity[0]),           # inlet velocity of the freely propagating flame = S_L
        "T_max": float(T.max()),
        "T_end": float(T[-1]),
        "thickness_mm": float((T[-1] - T[0]) / dTdx.max() * 1e3),
        "points": len(x),
        "domain_mm": float((x[-1] - x[0]) * 1e3),
        "ratio": refine.ratio, "slope": refine.slope, "curve": refine.curve,
        "seconds": seconds,
    }


def flame_speed(phi: float, T0: float = 300.0, P: float = P_ATM, refine: Refine = REFINE_EXAMPLE,
                width: float = 0.03, transport: str = "mixture-averaged") -> dict:
    """One flame solved from scratch (Cantera's automatic initial guess)."""
    f = _new_flame(phi, T0, P, width, transport)
    f.set_refine_criteria(ratio=refine.ratio, slope=refine.slope, curve=refine.curve)
    t = time.perf_counter()
    f.solve(loglevel=0, auto=True)
    return _flame_row(f, phi, refine, time.perf_counter() - t)


def flame_speed_sweep(phis, T0: float = 300.0, P: float = P_ATM, refine: Refine = REFINE_EXAMPLE,
                      width: float = 0.03, transport: str = "mixture-averaged") -> list[dict]:
    """S_L over phi. The flame nearest phi = 1 is solved from scratch; every other phi starts
    from that converged solution (same start for each, so the order of phis does not matter),
    with only the inlet composition changed, and the grid is refined again to the criteria."""
    phis = [float(p) for p in phis]
    start = min(phis, key=lambda p: abs(p - 1.0))
    f = _new_flame(start, T0, P, width, transport)
    f.set_refine_criteria(ratio=refine.ratio, slope=refine.slope, curve=refine.curve)
    t = time.perf_counter()
    f.solve(loglevel=0, auto=True)
    first = _flame_row(f, start, refine, time.perf_counter() - t)
    base = f.to_array()
    rows = {start: first}
    for phi in phis:
        if phi == start:
            continue
        f.from_array(base)
        f.inlet.Y = gas(phi, T0, P).Y
        t = time.perf_counter()
        f.solve(loglevel=0, refine_grid=True, auto=False)
        rows[phi] = _flame_row(f, phi, refine, time.perf_counter() - t)
    return [rows[p] for p in phis]


def grid_study(phi: float = 1.0, T0: float = 300.0, P: float = P_ATM, levels=GRID_STUDY,
               width: float = 0.03, transport: str = "mixture-averaged") -> list[dict]:
    """The same flame on successively tighter refinement criteria (each level continues
    from the previous solution, so the grid only gets finer)."""
    f = _new_flame(phi, T0, P, width, transport)
    rows = []
    for i, r in enumerate(levels):
        f.set_refine_criteria(ratio=r.ratio, slope=r.slope, curve=r.curve)
        t = time.perf_counter()
        if i == 0:
            f.solve(loglevel=0, auto=True)
        else:
            f.solve(loglevel=0, refine_grid=True, auto=False)
        rows.append(_flame_row(f, phi, r, time.perf_counter() - t))
    return rows


def richardson(rows: list[dict]) -> dict:
    """Extrapolate S_L to zero grid spacing from the last three grid-study levels, taking the
    mean spacing as 1/points. Returns the observed order and the extrapolated S_L."""
    (n1, s1), (n2, s2), (n3, s3) = [(r["points"], r["S_L"]) for r in rows[-3:]]
    # errors e_i = C * n_i^-p ; s_i = S + e_i. Solve with the last two differences.
    d21, d32 = s1 - s2, s2 - s3
    if d21 == 0 or d32 == 0 or (d21 > 0) != (d32 > 0):
        return {"order": float("nan"), "S_L_extrapolated": s3}
    p = np.log(abs(d21 / d32)) / np.log(n3 / n2) if n3 != n2 else float("nan")
    r = (n3 / n2) ** p
    return {"order": float(p), "S_L_extrapolated": float(s3 - d32 / (r - 1))}


# --- (c) ignition delay -----------------------------------------------------------------------

def ignition_delay(T0: float, P: float = P_ATM, phi: float = 1.0, t_end: float = 2.0) -> dict:
    """Ignition delay of a constant-pressure, adiabatic reactor: the time of the steepest
    temperature rise, found from the integrator's own (adaptive) time steps."""
    g = gas(phi, T0, P)
    r = ct.IdealGasConstPressureReactor(g, energy="on", clone=True)
    net = ct.ReactorNet([r])
    ts, Ts = [0.0], [r.T]
    t0 = time.perf_counter()
    while net.time < t_end:
        net.step()
        ts.append(net.time)
        Ts.append(r.T)
        if r.T > T0 + 1000 and Ts[-1] - Ts[-2] < 1e-3:   # burnt and settling
            break
    ts, Ts = np.array(ts), np.array(Ts)
    rate = np.diff(Ts) / np.diff(ts)
    i = int(np.argmax(rate))
    return {
        "T0": T0, "P_atm": P / P_ATM, "phi": phi,
        "tau_s": float(0.5 * (ts[i] + ts[i + 1])),
        "T_final": float(Ts[-1]),
        "steps": len(ts), "seconds": time.perf_counter() - t0,
    }


def arrhenius_fit(rows: list[dict]) -> dict:
    """Fit ln(tau) = ln(A) + (E/R) / T0. Returns E in kJ/mol and R^2 of the straight line."""
    x = 1.0 / np.array([r["T0"] for r in rows])
    y = np.log([r["tau_s"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"E_over_R_K": float(slope), "E_kJ_per_mol": float(slope * ct.gas_constant / 1e6),
            "ln_A": float(intercept), "r2": float(r2)}


# --- (d) the burnt gas as an OpenFOAM fluid ---------------------------------------------------

def product_properties(phi: float = 1.0, T0: float = 300.0, P: float = P_ATM,
                       T_duct: float = 1500.0) -> dict:
    """Density, dynamic and kinematic viscosity of the products, at the adiabatic flame
    temperature and after cooling to T_duct (re-equilibrated at T_duct; frozen also given)."""
    g = gas(phi, T0, P)
    g.equilibrate("HP")
    hot = {"T": g.T, "rho": g.density, "mu": g.viscosity, "nu": g.viscosity / g.density,
           "W": g.mean_molecular_weight}
    g.TP = T_duct, P
    frozen = {"T": g.T, "rho": g.density, "mu": g.viscosity, "nu": g.viscosity / g.density,
              "W": g.mean_molecular_weight}
    g.equilibrate("TP")
    duct = {"T": g.T, "rho": g.density, "mu": g.viscosity, "nu": g.viscosity / g.density,
            "W": g.mean_molecular_weight}
    return {"phi": phi, "P": P, "mechanism": MECH, "transport": g.transport_model,
            "flame": hot, "duct": duct, "duct_frozen": frozen}
