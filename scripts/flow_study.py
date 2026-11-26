"""Steps 2 and 3: the OpenFOAM channel on three blockMesh levels and a Gmsh mesh.
Writes flow.txt, flow_summary.json and flow.png to --out (default results/).

Needs OpenFOAM: natively (simpleFoam on PATH) or through Docker
(F2F_FOAM_IMAGE=opencfd/openfoam-run:2406). Reuses <F2F_WORK>/study.json when pytest has
already run the same cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f2f import foam, study  # noqa: E402
from f2f.channel import LEVELS, observed_order, poiseuille  # noqa: E402
from f2f.literature import DURST_2005_SOURCE  # noqa: E402
from f2f.plotstyle import AQUA, BLUE, ORANGE, REF, setup  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()
    if foam.foam_mode() is None and not (Path(args.work or study.default_work()) / "study.json").exists():
        sys.exit("OpenFOAM not available: set F2F_FOAM_IMAGE (Docker) or put simpleFoam on PATH")
    res = study.run_study(args.work)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ch, cases, props = res["channel"], res["cases"], res["products"]
    L = []
    say = L.append
    run = os.environ.get("GITHUB_RUN_ID")
    say(f"OpenFOAM v2406 ({res['foam_mode']}{', ' + res['foam_image'] if res['foam_image'] else ''}), "
        f"simpleFoam, steady laminar; written {time.strftime('%Y-%m-%d %H:%M')}"
        + (f" by GitHub Actions run {run}" if run else ""))
    d = props["duct"]
    say(f"fluid: burnt stoichiometric methane-air at {d['T']:.0f} K, 1 atm (Cantera, GRI-Mech 3.0): "
        f"rho={d['rho']:.5f} kg/m^3, mu={d['mu']:.4e} Pa s, nu={d['nu']:.4e} m^2/s")
    say(f"channel: h={ch['h']} m, L={ch['length']:.3f} m ({ch['length_h']:.0f} h), one cell thick; "
        f"Re = U h / nu = {ch['re']:.0f} (Re on D_h = 2h: {2 * ch['re']:.0f}); "
        f"U_mean={ch['u_mean']:.5f} m/s")
    say(f"exact developed flow: u_max/U = 1.5; dp/dx = -12 mu U / h^2 = {ch['dpdx_pa']:.4f} Pa/m "
        f"(kinematic {ch['dpdx_kinematic']:.4f} m/s^2); entrance length (Durst 2005) "
        f"{ch['entrance_length'] / ch['h']:.2f} h")
    say(f"total wall time for the four cases: {res['total_wall_s']:.0f} s\n")

    hdr = (f"{'case':18s} {'cells':>6s} {'iters':>6s} {'solve s':>7s} {'u_max/U':>8s} {'L2 prof':>8s} "
           f"{'dp/dx Pa/m':>10s} {'err':>7s} {'vs 1-D':>7s} {'L_e/h':>6s} {'Durst':>6s}")
    say(hdr)
    say("-" * len(hdr))
    for name, c in cases.items():
        pr, p, e, s = c["profile"], c["pressure"], c["entrance"], c["solver"]
        say(f"{name:18s} {c['mesh']['cells']:6d} {s['iterations']:6d} {s['clock_s']:7.0f} "
            f"{pr['umax_ratio']:8.4f} {pr['l2_rel'] * 100:7.3f}% {p['dpdx_pa']:10.4f} "
            f"{p['rel_error'] * 100:+6.2f}% {p['rel_error_vs_discrete_1d'] * 100:+6.3f}% "
            f"{e['L_e_over_h']:6.2f} {e['durst_over_h']:6.2f}")
    say("\nerr: against the exact -12 mu U / h^2. vs 1-D: against the 1-D discrete solution on the same "
        "ny cells\n(f2f.channel.discrete_poiseuille), i.e. what is left once the mesh error is accounted for.")
    c0 = next(iter(cases.values()))
    say(f"L2 prof: relative L2 error of the cell values at x = {c0['profile']['station_h']} h against "
        f"6U(y/h)(1-y/h). dp/dx: least-squares slope of the centreline p over "
        f"{c0['pressure']['fit_from_h']:g}-{c0['pressure']['fit_to_h']:g} h.")
    say(f"L_e: centreline velocity reaches 99% of its mean over x = {c0['entrance']['u_fd_from_h']:g}-"
        f"{c0['entrance']['u_fd_to_h']:g} h. Durst: {DURST_2005_SOURCE}.")
    say("local dp/dx along the centreline, % steeper than the developed (1-D discrete) value: " + "; ".join(
        f"{n}: " + ", ".join(f"{k} h {v:+.3f}" for k, v in cases[n]["pressure"]["local_excess_pct"].items())
        for n in ("blockMesh_medium",)))

    block = [f"blockMesh_{k}" for k in LEVELS]
    say("\nmesh convergence (blockMesh, h/dy = 10, 20, 40; dx = h/4, h/8, h/16):")
    for key, sub, label in (("pressure", "rel_error", "dp/dx error"), ("profile", "l2_rel", "profile L2"),
                            ("entrance", "rel_error_vs_durst", "L_e vs Durst")):
        errs = [abs(cases[n][key][sub]) for n in block]
        orders = observed_order(errs)
        say(f"  {label:13s} " + "  ".join(f"{e * 100:.4f}%" for e in errs) +
            "   observed order " + ", ".join(f"{o:.2f}" for o in orders))
    say("  1-D discrete prediction for dp/dx error: " + "  ".join(
        f"{(cases[n]['discrete_1d']['dpdx'] / ch['dpdx_kinematic'] - 1) * 100:+.4f}%" for n in block))

    b, g = cases["blockMesh_medium"], cases["gmsh_medium"]
    say("\nblockMesh vs Gmsh, medium level (240 x 20 cells each):")
    for label, key, sub in (("dp/dx", "pressure", "dpdx_kinematic"), ("u_max/U", "profile", "umax_ratio"),
                            ("profile L2", "profile", "l2_rel"), ("L_e", "entrance", "L_e_m"),
                            ("inlet p", "pressure", "p_inlet_kinematic")):
        vb, vg = b[key][sub], g[key][sub]
        say(f"  {label:11s} blockMesh {vb:.8g}  Gmsh {vg:.8g}  difference {(vg / vb - 1) * 100:+.2e}%")
    say(f"  gmshToFoam wrote patch types {g['gmsh']['patch_types_from_gmshToFoam']}; fixed to "
        f"inlet/outlet patch, walls wall, frontAndBack empty")
    say(f"  iterations: blockMesh {b['solver']['iterations']}, Gmsh {g['solver']['iterations']}")
    say(f"\nentrance pressure loss beyond the developed gradient, K = (p_in - |dp/dx| L) / (U^2/2): " +
        ", ".join(f"{n} {cases[n]['pressure']['entrance_loss_K']:.3f}" for n in cases))

    text = "\n".join(L) + "\n"
    print(text)
    (out / "flow.txt").write_text(text, encoding="utf-8")
    slim = {"channel": ch, "products": props, "foam_image": res["foam_image"],
            "total_wall_s": res["total_wall_s"],
            "cases": {n: {k: v for k, v in c.items() if k not in ("profiles", "centreline")}
                      for n, c in cases.items()}}
    (out / "flow_summary.json").write_text(json.dumps(slim, indent=1) + "\n")
    plot(out / "flow.png", res)


def plot(path: Path, res: dict):
    plt = setup()
    ch, cases = res["channel"], res["cases"]
    h, U = ch["h"], ch["u_mean"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))

    a = ax[0]
    c = cases["blockMesh_medium"]
    yy = np.linspace(0, h, 200)
    a.plot(poiseuille(yy, h, U) / U, yy / h, color=REF, lw=1, label="Poiseuille 6U(y/h)(1-y/h)")
    for key, color in (("1.03", ORANGE), ("5.03", AQUA), ("25.03", BLUE)):
        p = c["profiles"][key]
        a.plot(np.array(p["u"]) / U, np.array(p["y"]) / h, "o", ms=3.5, color=color, label=f"x = {key} h")
    a.set(xlabel="u / U_mean", ylabel="y / h", title="Profiles, blockMesh medium (cell values)")
    a.legend(loc="center left", fontsize=7)

    a = ax[1]
    for name, color in (("blockMesh_coarse", AQUA), ("blockMesh_fine", BLUE), ("gmsh_medium", ORANGE)):
        cl = cases[name]["centreline"]
        a.plot(np.array(cl["x"]) / h, np.array(cl["u"]) / U, color=color, lw=1.5 if "gmsh" in name else 2,
               ls="--" if "gmsh" in name else "-", label=name.replace("_", " "))
    a.axvline(ch["entrance_length"] / h, color=REF, lw=1, ls=":")
    a.annotate(f"Durst L_e = {ch['entrance_length'] / h:.2f} h", (ch["entrance_length"] / h, 1.2),
               xytext=(4, 0), textcoords="offset points", fontsize=7, color=REF)
    a.set(xlabel="x / h", ylabel="u_centre / U_mean",
          title=f"Centreline velocity, Re = {ch['re']:.0f} (y = h/2 is a face: 2-cell average)")
    a.legend(loc="lower right", fontsize=7)

    a = ax[2]
    ny = [m.ny for m in LEVELS.values()]
    block = [f"blockMesh_{k}" for k in LEVELS]
    for key, sub, color, label in (("pressure", "rel_error", BLUE, "dp/dx error"),
                                   ("profile", "l2_rel", ORANGE, "profile L2 error")):
        a.loglog(ny, [abs(cases[n][key][sub]) for n in block], "o-", color=color, label=label)
    e0 = abs(cases[block[0]]["pressure"]["rel_error"])
    a.loglog(ny, [e0 * (ny[0] / n) ** 2 for n in ny], color=REF, lw=1, ls=":", label="slope -2")
    a.set_xticks(ny)
    a.set_xticklabels([str(n) for n in ny])
    a.minorticks_off()
    a.set(xlabel="cells across the channel (log scale)", ylabel="relative error",
          title="Mesh convergence (blockMesh)")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
