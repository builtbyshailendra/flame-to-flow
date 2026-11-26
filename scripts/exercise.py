"""Run one channel case with your own settings and print the checks (LEARN.md, "Your turn").

    python scripts/exercise.py --re 400                       # blockMesh, medium level
    python scripts/exercise.py --mesher gmsh --bump 0.2       # Gmsh, cells clustered at the walls
    python scripts/exercise.py --nx 480 --ny 20               # your own blockMesh counts

Needs OpenFOAM (CI has it: put the arguments in exercise.txt on a branch and push; the flow job
runs `python scripts/exercise.py $(cat exercise.txt)` and prints the result in its log).
Without OpenFOAM it still writes the case and the mesh, so you can read them.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f2f import foam, gmsh_channel, study  # noqa: E402
from f2f.channel import LEVELS, Mesh, discrete_poiseuille_faces, poiseuille  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=200.0, help="Re = U h / nu (200 in the study)")
    ap.add_argument("--mesher", choices=["blockMesh", "gmsh"], default="blockMesh")
    ap.add_argument("--level", choices=list(LEVELS), default="medium")
    ap.add_argument("--nx", type=int)
    ap.add_argument("--ny", type=int)
    ap.add_argument("--bump", type=float, default=None, help="Gmsh only: Bump coefficient < 1")
    ap.add_argument("--length-h", type=float, default=30.0, help="channel length in heights (30)")
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args()

    work = Path(a.work or study.default_work())
    work.mkdir(parents=True, exist_ok=True)
    ch, props = study.channel_from_cantera(a.re)
    ch = dataclasses.replace(ch, length_h=a.length_h)
    lv = LEVELS[a.level]
    mesh = Mesh(a.nx or lv.nx, a.ny or lv.ny)
    name = (f"exercise_{a.mesher}_re{a.re:g}_L{a.length_h:g}_{mesh.nx}x{mesh.ny}"
            + (f"_bump{a.bump:g}" if a.bump else ""))
    print(f"case {name}: Re={a.re:g}, U={ch.u_mean:.4f} m/s, nu={ch.nu:.4e} m^2/s, L={a.length_h:g} h, "
          f"{mesh.nx} x {mesh.ny} cells, {a.mesher}" + (f", bump {a.bump}" if a.bump else ""))
    print(f"known answers: u_max/U = 1.5, dp/dx = {ch.dpdx_kinematic:.4f} m/s^2 (kinematic) = "
          f"{ch.dpdx_pa:.4f} Pa/m, entrance length (Durst 2005) {ch.entrance_length / ch.h:.2f} h")

    if foam.foam_mode() is None:
        case = foam.write_case(work / name, ch, mesh, props=props["duct"])
        if a.mesher == "gmsh":
            gmsh_channel.build(ch, mesh, case / "channel.msh", wall_bump=a.bump)
        print(f"OpenFOAM not available: case written to {case} (read it; run it in CI)")
        return
    if a.mesher == "gmsh":
        res = study.run_gmsh(work, name, ch, mesh, props["duct"], wall_bump=a.bump)
    else:
        res = study.run_block(work, name, ch, mesh, props["duct"])

    s, pr, p, e = res["solver"], res["profile"], res["pressure"], res["entrance"]
    print(f"checkMesh ok: {res['checkMesh']['ok']}; simpleFoam converged: {s['converged']} in "
          f"{s['iterations']} iterations ({s['clock_s']:.0f} s)")
    print(f"u_max/U (parabola fit at x = {pr['station_h']} h): {pr['umax_ratio']:.4f}")
    print(f"profile L2 error: {pr['l2_rel'] * 100:.3f}%")
    print(f"dp/dx (fit over {p['fit_from_h']:g}-{p['fit_to_h']:g} h): {p['dpdx_kinematic']:.4f} m/s^2 = "
          f"{p['dpdx_pa']:.4f} Pa/m, error {p['rel_error'] * 100:+.3f}%")
    print("local dp/dx, % steeper than developed: " +
          ", ".join(f"{k} h {v:+.3f}" for k, v in p["local_excess_pct"].items()))
    print(f"entrance length: {e['L_e_over_h']:.2f} h (Durst {e['durst_over_h']:.2f} h, "
          f"{e['rel_error_vs_durst'] * 100:+.1f}%)")
    # what the 1-D discrete equations predict on the same cells across the channel
    if a.mesher == "gmsh":
        import meshio
        yf = np.unique(np.round(meshio.read(work / name / "channel.msh").points[:, 1], 12))
    else:
        yf = np.linspace(0, ch.h, mesh.ny + 1)
    yc, uc, g = discrete_poiseuille_faces(yf, ch.u_mean, ch.nu)
    ex = poiseuille(yc, ch.h, ch.u_mean)
    print(f"cell height at the wall {np.diff(yf)[0] / ch.h:.4f} h, at the centre {np.diff(yf).max() / ch.h:.4f} h")
    print(f"1-D discrete prediction on these cells: dp/dx error {(g / ch.dpdx_kinematic - 1) * 100:+.3f}%, "
          f"profile L2 {np.sqrt(np.mean((uc - ex) ** 2) / np.mean(ex ** 2)) * 100:.3f}%")


if __name__ == "__main__":
    main()
