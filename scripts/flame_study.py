"""Step 1: the Cantera part, full local sweep. Writes results/flame_*.csv, products.json,
flame.txt and flame.png.

    python scripts/flame_study.py            # about 10 minutes on a laptop
    python scripts/flame_study.py --quick    # coarse grid, three phis (what CI checks)
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import cantera as ct
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f2f import flame, literature  # noqa: E402
from f2f.plotstyle import AQUA, BLUE, ORANGE, REF, setup  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def write_csv(path: Path, rows: list[dict], fmt: dict | None = None):
    fmt = fmt or {}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt[k].format(v) if k in fmt else
                            (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    t_all = time.perf_counter()
    say(f"Cantera {ct.__version__}, {flame.MECH}, mixture-averaged transport; "
        f"Python {platform.python_version()} on {platform.system()} ({platform.machine()})")
    say(f"run: {'quick (CI-size)' if args.quick else 'full local sweep'}, "
        f"{time.strftime('%Y-%m-%d %H:%M')}")

    # (a) adiabatic flame temperature
    phis_t = np.round(np.arange(0.6, 1.401, 0.05), 2)
    t = time.perf_counter()
    temps = flame.flame_temperature_sweep(phis_t)
    say(f"\n(a) Adiabatic flame temperature, 300 K, 1 atm, equilibrate('HP'): "
        f"{len(temps)} mixtures in {time.perf_counter() - t:.2f} s")
    peak = max(temps, key=lambda r: r["T_ad"])
    stoich = next(r for r in temps if abs(r["phi"] - 1) < 1e-9)
    say(f"    phi=1.00: {stoich['T_ad']:.1f} K; peak {peak['T_ad']:.1f} K at phi={peak['phi']:.2f}; "
        f"phi=0.60: {temps[0]['T_ad']:.1f} K; phi=1.40: {temps[-1]['T_ad']:.1f} K")
    write_csv(out / "flame_temperature.csv", temps)

    # (b) laminar flame speed: grid study, then the sweep
    if args.quick:
        levels = flame.GRID_STUDY[:2]
        phis_s = [0.8, 1.0, 1.2]
        sweep_refine = flame.REFINE_EXAMPLE
    else:
        levels = flame.GRID_STUDY
        phis_s = [float(p) for p in np.round(np.arange(0.7, 1.301, 0.05), 2)]
        sweep_refine = flame.REFINE_SWEEP
    t = time.perf_counter()
    grid = flame.grid_study(1.0, levels=levels)
    say(f"\n(b) Laminar flame speed, FreeFlame, 300 K, 1 atm. Grid study at phi=1 "
        f"({time.perf_counter() - t:.0f} s; each level continues from the previous one):")
    for r in grid:
        say(f"    {flame.Refine(r['ratio'], r['slope'], r['curve']).label():34s} "
            f"{r['points']:5d} points  S_L={r['S_L']:.5f} m/s  thickness {r['thickness_mm']:.3f} mm"
            f"  {r['seconds']:.1f} s")
    rich = flame.richardson(grid) if len(grid) >= 3 else {}
    if rich:
        say(f"    Richardson estimate from the last three levels: S_L = "
            f"{rich['S_L_extrapolated']:.5f} m/s, observed order {rich['order']:.2f} in 1/points")
    write_csv(out / ("flame_speed_grid_quick.csv" if args.quick else "flame_speed_grid.csv"), grid)

    t = time.perf_counter()
    speeds = flame.flame_speed_sweep(phis_s, refine=sweep_refine)
    say(f"\n    Sweep with {sweep_refine.label()}: {len(speeds)} flames in "
        f"{time.perf_counter() - t:.0f} s (phi nearest 1 from scratch, the others continued from it)")
    for r in speeds:
        try:
            exp = literature.hermanns_speed(r["phi"])
        except KeyError:
            exp = float("nan")
        r["S_L_exp_298K"] = exp
        r["diff_pct"] = (r["S_L"] / exp - 1) * 100
        say(f"    phi={r['phi']:.2f}  S_L={r['S_L']:.4f} m/s  measured(298 K)={exp:.3f}  "
            f"{r['diff_pct']:+5.1f}%  T_max={r['T_max']:.0f} K  {r['points']} pts  {r['seconds']:.1f} s")
    best = max(speeds, key=lambda r: r["S_L"])
    say(f"    peak in this sweep: {best['S_L']:.4f} m/s at phi={best['phi']:.2f} "
        f"(measured peak 0.372 m/s at phi=1.05)")
    write_csv(out / ("flame_speed_quick.csv" if args.quick else "flame_speed.csv"), speeds)

    # the measurements are at 298 K, the sweep at 300 K: how much do 2 K matter?
    t = time.perf_counter()
    s298 = flame.flame_speed_sweep([1.0], T0=298.0, refine=sweep_refine)[0]
    s300 = next(r for r in speeds if abs(r["phi"] - 1) < 1e-9)
    say(f"    phi=1 at 298 K: S_L={s298['S_L']:.4f} m/s ({(s298['S_L'] / s300['S_L'] - 1) * 100:+.2f}% "
        f"vs 300 K; {time.perf_counter() - t:.0f} s); vs measured 0.364: "
        f"{(s298['S_L'] / 0.364 - 1) * 100:+.1f}%")
    say(f"    source: {literature.HERMANNS_2007_SOURCE}")

    # (c) ignition delay
    t = time.perf_counter()
    ign = []
    temps0 = [1200, 1300, 1400, 1500, 1600, 1700, 1800]
    for p_atm in (1.0, 10.0):
        for T0 in temps0:
            ign.append(flame.ignition_delay(T0, P=p_atm * ct.one_atm))
    say(f"\n(c) Ignition delay, constant-pressure reactor, phi=1, max dT/dt "
        f"({len(ign)} runs, {time.perf_counter() - t:.1f} s):")
    fits = {}
    for p_atm in (1.0, 10.0):
        rows = [r for r in ign if r["P_atm"] == p_atm]
        fit = flame.arrhenius_fit(rows)
        fits[p_atm] = fit
        say(f"    {p_atm:4.0f} atm: " + ", ".join(f"{r['T0']:.0f} K {r['tau_s'] * 1e3:.4g} ms" for r in rows))
        say(f"             ln(tau) vs 1/T0: E/R = {fit['E_over_R_K']:.0f} K, "
            f"E = {fit['E_kJ_per_mol']:.0f} kJ/mol, R^2 = {fit['r2']:.5f}")
    ratio = [a["tau_s"] / b["tau_s"] for a, b in zip([r for r in ign if r["P_atm"] == 1.0],
                                                      [r for r in ign if r["P_atm"] == 10.0])]
    say(f"    1 atm / 10 atm delay ratio: {min(ratio):.1f} to {max(ratio):.1f} "
        f"(tau ~ P^-n with n = {np.log10(min(ratio)):.2f} to {np.log10(max(ratio)):.2f})")
    write_csv(out / "ignition_delay.csv", ign)

    # (d) the OpenFOAM inputs
    props = flame.product_properties(phi=1.0, T_duct=1500.0)
    (out / "products.json").write_text(json.dumps(props, indent=1) + "\n")
    say("\n(d) Burnt gas, phi=1, 1 atm (the OpenFOAM inputs):")
    for k, label in (("flame", "at T_ad"), ("duct", "cooled, re-equilibrated"),
                     ("duct_frozen", "cooled, frozen composition")):
        d = props[k]
        say(f"    {label:28s} T={d['T']:.1f} K  rho={d['rho']:.5f} kg/m^3  mu={d['mu']:.4e} Pa s  "
            f"nu={d['nu']:.4e} m^2/s")
    say(f"\ntotal {time.perf_counter() - t_all:.0f} s")

    name = "flame_quick" if args.quick else "flame"
    (out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(out / f"{name}.png", temps, speeds, grid, ign, fits)


def plot(path, temps, speeds, grid, ign, fits):
    plt = setup()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))
    a = ax[0, 0]
    a.plot([r["phi"] for r in temps], [r["T_ad"] for r in temps], color=BLUE)
    a.set(xlabel="equivalence ratio", ylabel="K", title="Adiabatic flame temperature (equilibrate HP)")
    a = ax[0, 1]
    a.plot([r["phi"] for r in speeds], [r["S_L"] for r in speeds], "-o", color=BLUE,
           label="Cantera, GRI-Mech 3.0, 300 K")
    ph = [p for p, s, e in literature.HERMANNS_2007]
    a.errorbar(ph, [s / 100 for p, s, e in literature.HERMANNS_2007],
               yerr=[e / 100 for p, s, e in literature.HERMANNS_2007], fmt="s", ms=4,
               color=REF, mfc="none", label="measured, heat flux, 298 K (Hermanns 2007)")
    a.set(xlabel="equivalence ratio", ylabel="m/s", title="Laminar flame speed", xlim=(0.6, 1.55))
    a.legend(loc="lower center")
    a = ax[1, 0]
    for p_atm, color in ((1.0, BLUE), (10.0, ORANGE)):
        rows = [r for r in ign if r["P_atm"] == p_atm]
        x = [1000 / r["T0"] for r in rows]
        a.semilogy(x, [r["tau_s"] * 1e3 for r in rows], "o", color=color,
                   label=f"{p_atm:.0f} atm (E = {fits[p_atm]['E_kJ_per_mol']:.0f} kJ/mol)")
        f = fits[p_atm]
        xx = np.linspace(min(x), max(x), 50)
        a.semilogy(xx, 1e3 * np.exp(f["ln_A"] + f["E_over_R_K"] * xx / 1000),
                   "-", lw=1, color=color)
    a.set(xlabel="1000 / T0  (1/K)", ylabel="ms", title="Ignition delay, phi = 1 (straight line: Arrhenius fit)")
    a.legend()
    a = ax[1, 1]
    a.semilogx([r["points"] for r in grid], [r["S_L"] for r in grid], "-o", color=AQUA)
    for r in grid:
        a.annotate(f"slope {r['slope']:g}", (r["points"], r["S_L"]), textcoords="offset points",
                   xytext=(4, 4), fontsize=7, color=REF)
    a.set(xlabel="grid points", ylabel="m/s", title="Flame speed at phi = 1 vs grid refinement")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
