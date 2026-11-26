"""Building, running and reading OpenFOAM cases (OpenFOAM v2406, openfoam.com).

OpenFOAM runs natively when `simpleFoam` is on PATH, or through Docker when the environment
variable F2F_FOAM_IMAGE names an image (CI uses opencfd/openfoam-run:2406). Otherwise the
functions that need it raise FoamMissing and the tests that need it are skipped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .channel import Channel, Mesh

TEMPLATE = Path(__file__).resolve().parent.parent / "cases" / "channel"

# x stations (in channel heights) of the profiles sampled across the channel. The 0.03 offset
# keeps every station off the cell faces of all three mesh levels (dx = h/4, h/8, h/16).
STATIONS_H = (1.03, 2.53, 5.03, 10.03, 20.03, 25.03)


class FoamMissing(RuntimeError):
    pass


class FoamFailed(RuntimeError):
    pass


def foam_mode() -> str | None:
    if shutil.which("simpleFoam"):
        return "native"
    if os.environ.get("F2F_FOAM_IMAGE") and shutil.which("docker"):
        return "docker"
    return None


def station_name(x_h: float) -> str:
    return "x" + f"{x_h:.2f}".replace(".", "p")


def write_case(case: Path, ch: Channel, mesh: Mesh, *, end_time: int = 20000,
               p_tol: float = 1e-6, u_tol: float = 1e-7, props: dict | None = None,
               centre_points: int = 1201) -> Path:
    """Copy the template case and fill in its {{placeholders}}."""
    case = Path(case)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(TEMPLATE, case)
    z_mid = ch.dz / 2
    eps = 1e-6 * ch.h
    sets = []
    for s in STATIONS_H:
        x = s * ch.h
        sets.append(
            f"    {station_name(s)}\n    {{\n        type    midPoint;\n        axis    xyz;\n"
            f"        start   ({x:.10g} {eps:.10g} {z_mid:.10g});\n"
            f"        end     ({x:.10g} {ch.h - eps:.10g} {z_mid:.10g});\n    }}"
        )
    props = props or {}
    values = {
        "L": f"{ch.length:.10g}", "h": f"{ch.h:.10g}", "dz": f"{ch.dz:.10g}",
        "nx": str(mesh.nx), "ny": str(mesh.ny),
        "endTime": str(end_time), "pTol": f"{p_tol:g}", "UTol": f"{u_tol:g}",
        "nu": f"{ch.nu:.10g}", "U": f"{ch.u_mean:.10g}", "Re": f"{ch.re:g}",
        "mu": f"{props.get('mu', ch.nu * ch.rho):.6g}", "rho": f"{props.get('rho', ch.rho):.6g}",
        "Tduct": f"{props.get('T', 1500):g}",
        "hHalf": f"{ch.h / 2:.10g}", "zMid": f"{z_mid:.10g}", "nCentre": str(centre_points),
        "profileSets": "\n".join(sets),
    }
    for p in case.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            text = re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), text)
            p.write_text(text, encoding="utf-8", newline="\n")
    return case


def run(case: Path, commands: list[tuple[str, str]], timeout: int = 1800) -> None:
    """Run OpenFOAM commands in the case directory, each as (log name, command line), writing
    log.<name>. Stops at the first failure."""
    mode = foam_mode()
    if mode is None:
        raise FoamMissing("OpenFOAM not found (no simpleFoam on PATH and F2F_FOAM_IMAGE unset)")
    case = Path(case).resolve()
    lines = [f"{cmd} > log.{name} 2>&1" for name, cmd in commands]
    script = "set -e\n" + "\n".join(lines) + "\n"
    if mode == "native":
        argv = ["bash", "-c", script]
    else:
        image = os.environ["F2F_FOAM_IMAGE"]
        # The image's OpenFOAM environment is not loaded in a plain `bash -c`: source it first.
        env = ("for f in /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc; do "
               "[ -f \"$f\" ] && . \"$f\" && break; done\n")
        mount = str(case.parent)
        argv = ["docker", "run", "--rm", "--entrypoint", "bash",
                "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                "-v", f"{mount}:{mount}", "-w", str(case), image, "-c", env + script]
    proc = subprocess.run(argv, cwd=case, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tails = []
        for name, _ in commands:
            log = case / f"log.{name}"
            if log.exists():
                tails.append(f"--- log.{name} (last lines)\n" + "\n".join(
                    log.read_text(errors="replace").splitlines()[-25:]))
        raise FoamFailed(f"OpenFOAM failed in {case} (exit {proc.returncode})\n"
                         f"{proc.stdout[-2000:]}{proc.stderr[-2000:]}\n" + "\n".join(tails))


def fix_boundary(boundary: Path, types: dict[str, str]) -> dict[str, str]:
    """Set patch types in constant/polyMesh/boundary (gmshToFoam makes every patch a plain
    'patch'). Returns the types found before the change. Raises if a patch is missing."""
    boundary = Path(boundary)
    text = boundary.read_text(encoding="utf-8")
    before = {}
    for name, new_type in types.items():
        m = re.search(r"(^\s*" + re.escape(name) + r"\s*\{)(.*?)(\})", text, re.S | re.M)
        if not m:
            raise KeyError(f"patch {name} not in {boundary}")
        body = m.group(2)
        old = re.search(r"\btype\s+(\w+)\s*;", body)
        before[name] = old.group(1) if old else ""
        body = re.sub(r"\btype\s+\w+\s*;", f"type            {new_type};", body, count=1)
        # gmshToFoam may also write physicalType and inGroups; let OpenFOAM derive the groups
        body = re.sub(r"\n\s*physicalType\s+\w+\s*;", "", body)
        body = re.sub(r"\n\s*inGroups\s+[^;]*;", "", body)
        text = text[:m.start(2)] + body + text[m.end(2):]
    boundary.write_text(text, encoding="utf-8", newline="\n")
    return before


def latest_time(case: Path) -> str:
    times = []
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                times.append((float(d.name), d.name))
            except ValueError:
                pass
    return max(times)[1]


def read_set(case: Path, func: str, set_name: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one field of one sampled set written by setFormat raw (axis xyz).
    Returns (xyz with shape (n, 3), values with shape (n,) or (n, 3))."""
    base = Path(case) / "postProcessing" / func
    tdir = base / latest_time(base)
    for f in sorted(tdir.glob(f"{set_name}_*.xy")):
        fields = f.stem[len(set_name) + 1:].split("_")
        if field not in fields:
            continue
        data = np.loadtxt(f, comments="#", ndmin=2)
        # columns: x y z, then the fields in file-name order. v2406 writes all fields of a set
        # into one file (centreline_p_U.xy: x y z p U_x U_y U_z), so the column count alone
        # can't say where a field starts: use each field's number of components.
        start = 3
        for name in fields:
            n = FIELD_COMPONENTS[name]
            if name == field:
                vals = data[:, start:start + n]
                return data[:, :3], (vals[:, 0] if n == 1 else vals)
            start += n
    raise FileNotFoundError(f"no {field} for set {set_name} in {tdir}: "
                            f"{[p.name for p in tdir.iterdir()]}")


FIELD_COMPONENTS = {"p": 1, "U": 3}


_TIME = re.compile(r"^Time = (\S+)", re.M)
_RES = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")
_CONV = re.compile(r"SIMPLE solution converged in (\S+) iterations")
_EXEC = re.compile(r"ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.eE+-]+) s")


def read_solver_log(log: Path) -> dict:
    text = Path(log).read_text(errors="replace")
    times = _TIME.findall(text)
    conv = _CONV.search(text)
    execs = _EXEC.findall(text)
    last = {}
    tail = text[text.rfind("\nTime = "):] if "\nTime = " in text else text
    for name, value in _RES.findall(tail):
        last.setdefault(name, float(value))
    return {
        "iterations": int(float(times[-1])) if times else 0,
        "converged": bool(conv),
        "final_initial_residuals": last,
        "execution_s": float(execs[-1][0]) if execs else float("nan"),
        "clock_s": float(execs[-1][1]) if execs else float("nan"),
    }


def read_checkmesh(log: Path) -> dict:
    text = Path(log).read_text(errors="replace")
    def num(label):
        m = re.search(r"^\s*" + label + r":\s+(\d+)", text, re.M)
        return int(m.group(1)) if m else None
    patches = {}
    for m in re.finditer(r"^\s{4}(\w+)\s+(\d+)\s+(\d+)\s+ok", text, re.M):
        patches[m.group(1)] = int(m.group(2))
    return {"ok": "Mesh OK." in text, "cells": num("cells"), "faces": num("faces"),
            "points": num("points"), "patch_faces": patches}

