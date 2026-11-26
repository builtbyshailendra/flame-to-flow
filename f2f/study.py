"""The OpenFOAM part end to end: Cantera properties -> cases -> runs -> checks.

run_study() builds and runs the channel on three blockMesh levels and on a Gmsh mesh of the
medium level, samples each with postProcess, and returns every metric the tests and the
README use. Results are cached in <work>/study.json, keyed by the inputs, so pytest and the
report script can share one set of runs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import foam, gmsh_channel
from .channel import (LEVELS, Channel, Mesh, discrete_poiseuille, entrance_length,
                      pressure_gradient, profile_metrics)
from .flame import product_properties

# Where the fully developed checks are made, in channel heights back from the outlet (in the
# 30 h channel: the profile at 25.03 h, the pressure gradient fitted over 22-28 h, the
# developed centreline velocity averaged over 20-27 h). The pressure gradient first used
# 15-27 h and came out 0.1% steeper than the discrete answer on every mesh: at 15 h the local
# gradient was still 0.6% steeper (0.09% at 20 h). It develops more slowly than the
# centreline velocity, which is within 1% by 9 h.
FIT_BACK_H = (8.0, 2.0)
UFD_BACK_H = (10.0, 3.0)

SAMPLE_CMDS = [
    ("postProcess.sample", "postProcess -func sample -latestTime"),
    ("postProcess.centreline", "postProcess -func sampleCentreline -latestTime"),
]


def default_work() -> Path:
    return Path(os.environ.get("F2F_WORK", Path(__file__).resolve().parent.parent / "work"))


def channel_from_cantera(re: float = 200.0, T_duct: float = 1500.0) -> tuple[Channel, dict]:
    props = product_properties(phi=1.0, T_duct=T_duct)
    duct = props["duct"]
    return Channel(nu=duct["nu"], rho=duct["rho"], re=re), props


def analyse(case: Path, ch: Channel, mesh: Mesh) -> dict:
    """Every check for one finished case."""
    h, U = ch.h, ch.u_mean
    out = {"mesh": {"nx": mesh.nx, "ny": mesh.ny, "cells": mesh.cells}}
    out["solver"] = foam.read_solver_log(case / "log.simpleFoam")
    out["checkMesh"] = foam.read_checkmesh(case / "log.checkMesh")

    profiles = {}
    for s in foam.stations(ch):
        xyz, u = foam.read_set(case, "sample", foam.station_name(s), "U")
        profiles[s] = {"y": xyz[:, 1].tolist(), "u": u[:, 0].tolist(), "v": u[:, 1].tolist()}
    fd_station = foam.developed_station(ch)
    fd = profiles[fd_station]
    out["profile"] = profile_metrics(np.array(fd["y"]), np.array(fd["u"]), h, U)
    out["profile"]["station_h"] = fd_station
    out["profile"]["v_max_rel"] = float(np.max(np.abs(fd["v"])) / U)

    # what the 1-D discrete equations predict on this many cells (developed flow)
    y1, u1, g1 = discrete_poiseuille(mesh.ny, h, U, ch.nu)
    out["discrete_1d"] = {"dpdx": float(g1), "umax_cell_ratio": float(u1.max() / U),
                          "l2_rel": profile_metrics(y1, u1, h, U)["l2_rel"]}

    xyz, uc = foam.read_set(case, "sampleCentreline", "centreline", "U")
    _, pc = foam.read_set(case, "sampleCentreline", "centreline", "p")
    x = xyz[:, 0]
    Lh = ch.length_h
    fit = (Lh - FIT_BACK_H[0], Lh - FIT_BACK_H[1])
    dp = pressure_gradient(x, pc, fit[0] * h, fit[1] * h)
    out["pressure"] = {
        "dpdx_kinematic": dp["dpdx"], "dpdx_exact": ch.dpdx_kinematic,
        "rel_error": dp["dpdx"] / ch.dpdx_kinematic - 1.0,
        "rel_error_vs_discrete_1d": dp["dpdx"] / g1 - 1.0,
        "dpdx_pa": dp["dpdx"] * ch.rho, "dpdx_pa_exact": ch.dpdx_pa,
        "fit_from_h": fit[0], "fit_to_h": fit[1],
        "fit_points": dp["points"], "fit_max_resid_rel": dp["max_resid_rel"],
        "p_inlet_kinematic": float(pc[0]), "p_inlet_pa": float(pc[0] * ch.rho),
        # pressure drop beyond the fully developed gradient over the length: the entrance loss
        "entrance_loss_K": float((pc[0] - (-ch.dpdx_kinematic) * ch.length) / (0.5 * U ** 2)),
    }
    ufd = (Lh - UFD_BACK_H[0], Lh - UFD_BACK_H[1])
    le = entrance_length(x, uc[:, 0], ufd[0] * h, ufd[1] * h)
    out["entrance"] = {"L_e_m": le["L_e"], "L_e_over_h": le["L_e"] / h,
                       "durst_over_h": ch.entrance_length / h,
                       "rel_error_vs_durst": le["L_e"] / ch.entrance_length - 1.0,
                       "uc_fd_ratio": le["u_fd"] / U, "u_fd_from_h": ufd[0], "u_fd_to_h": ufd[1]}
    grad = np.gradient(pc, x)
    out["pressure"]["local_excess_pct"] = {
        f"{xh:g}": float((np.interp(xh * h, x, grad) / g1 - 1) * 100) for xh in (5, 10, 15, 20, 25)
        if xh < Lh}
    out["centreline"] = {"x": x[::4].tolist(), "u": uc[::4, 0].tolist(), "p": pc[::4].tolist()}
    out["profiles"] = {f"{k:.2f}": v for k, v in profiles.items()}
    return out


def run_block(work: Path, name: str, ch: Channel, mesh: Mesh, props: dict, **case_kw) -> dict:
    case = foam.write_case(work / name, ch, mesh, props=props, **case_kw)
    t = time.perf_counter()
    foam.run(case, [("blockMesh", "blockMesh"), ("checkMesh", "checkMesh"),
                    ("simpleFoam", "simpleFoam"), *SAMPLE_CMDS])
    res = analyse(case, ch, mesh)
    res["wall_s"] = time.perf_counter() - t
    res["mesher"] = "blockMesh"
    return res


def run_gmsh(work: Path, name: str, ch: Channel, mesh: Mesh, props: dict,
             wall_bump: float | None = None, **case_kw) -> dict:
    case = foam.write_case(work / name, ch, mesh, props=props, **case_kw)
    (case / "system" / "blockMeshDict").unlink()            # this case's mesh comes from Gmsh
    t = time.perf_counter()
    info = gmsh_channel.build(ch, mesh, case / "channel.msh", wall_bump=wall_bump)
    foam.run(case, [("gmshToFoam", "gmshToFoam channel.msh")])
    before = foam.fix_boundary(case / "constant" / "polyMesh" / "boundary",
                               {"inlet": "patch", "outlet": "patch", "walls": "wall",
                                "frontAndBack": "empty"})
    foam.run(case, [("checkMesh", "checkMesh"), ("simpleFoam", "simpleFoam"), *SAMPLE_CMDS])
    res = analyse(case, ch, mesh)
    res["wall_s"] = time.perf_counter() - t
    res["mesher"] = "gmsh"
    res["gmsh"] = {"elements": info["volume_elements"], "nodes": info["nodes"],
                   "patch_types_from_gmshToFoam": before, "wall_bump": wall_bump}
    return res


def run_study(work: Path | None = None, re: float = 200.0, force: bool = False) -> dict:
    work = Path(work or default_work())
    work.mkdir(parents=True, exist_ok=True)
    ch, props = channel_from_cantera(re)
    key = {"channel": ch.as_dict(), "levels": {k: [m.nx, m.ny] for k, m in LEVELS.items()},
           "template": _template_hash()}
    cache = work / "study.json"
    if cache.exists() and not force:
        old = json.loads(cache.read_text())
        if old.get("key") == key:
            return old
    t = time.perf_counter()
    cases = {}
    for level, mesh in LEVELS.items():
        cases[f"blockMesh_{level}"] = run_block(work, f"block_{level}", ch, mesh, props["duct"])
    cases["gmsh_medium"] = run_gmsh(work, "gmsh_medium", ch, LEVELS["medium"], props["duct"])
    result = {"key": key, "channel": ch.as_dict(), "products": props, "cases": cases,
              "total_wall_s": time.perf_counter() - t, "foam_mode": foam.foam_mode(),
              "foam_image": os.environ.get("F2F_FOAM_IMAGE", "")}
    cache.write_text(json.dumps(result, indent=1))
    return result


def _template_hash() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(foam.TEMPLATE.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(foam.TEMPLATE).as_posix().encode() + p.read_bytes())
    for mod in (foam, gmsh_channel):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]
