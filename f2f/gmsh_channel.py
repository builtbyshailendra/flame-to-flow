"""The channel mesh made a second, independent way: the Gmsh Python API.

Structured (transfinite) quads in the x-y plane, extruded one layer in z to hexahedra, with
physical groups named like the blockMesh patches, written as .msh version 2.2 ASCII (the
format OpenFOAM's gmshToFoam reads).
"""
from __future__ import annotations

from pathlib import Path

import gmsh

from .channel import Channel, Mesh

PATCHES = ("inlet", "outlet", "walls", "frontAndBack")


def build(ch: Channel, mesh: Mesh, out: Path, wall_bump: float | None = None) -> dict:
    """Write the mesh to `out` and return counts read back from Gmsh.

    wall_bump: None for uniform cells across the channel; a Gmsh "Bump" coefficient below 1
    (e.g. 0.2) clusters the cells towards both walls.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    L, h, dz = ch.length, ch.h, ch.dz
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("channel")
        geo = gmsh.model.geo
        p1 = geo.addPoint(0, 0, 0)
        p2 = geo.addPoint(L, 0, 0)
        p3 = geo.addPoint(L, h, 0)
        p4 = geo.addPoint(0, h, 0)
        bottom = geo.addLine(p1, p2)
        right = geo.addLine(p2, p3)
        top = geo.addLine(p3, p4)
        left = geo.addLine(p4, p1)
        surface = geo.addPlaneSurface([geo.addCurveLoop([bottom, right, top, left])])

        # transfinite: a fixed number of nodes on each curve -> a structured grid
        geo.mesh.setTransfiniteCurve(bottom, mesh.nx + 1)
        geo.mesh.setTransfiniteCurve(top, mesh.nx + 1)
        for c in (right, left):
            if wall_bump is None:
                geo.mesh.setTransfiniteCurve(c, mesh.ny + 1)
            else:
                geo.mesh.setTransfiniteCurve(c, mesh.ny + 1, "Bump", wall_bump)
        geo.mesh.setTransfiniteSurface(surface)
        geo.mesh.setRecombine(2, surface)          # triangles -> quads

        # one layer of hexes: numElements=[1], recombine=True (prisms otherwise)
        ext = geo.extrude([(2, surface)], 0, 0, dz, numElements=[1], recombine=True)
        geo.synchronize()

        volume = [t for d, t in ext if d == 3]
        groups = {name: [] for name in PATCHES}
        eps = 1e-9 * L
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if xmax < eps:
                groups["inlet"].append(tag)
            elif xmin > L - eps:
                groups["outlet"].append(tag)
            elif ymax < eps or ymin > h - eps:
                groups["walls"].append(tag)
            elif zmax < eps or zmin > dz - eps:
                groups["frontAndBack"].append(tag)
        for name in PATCHES:
            gmsh.model.addPhysicalGroup(2, groups[name], name=name)
        gmsh.model.addPhysicalGroup(3, volume, name="fluid")

        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(out))

        counts = {}
        for etype in gmsh.model.mesh.getElementTypes(3):
            name = gmsh.model.mesh.getElementProperties(etype)[0]
            counts[name] = len(gmsh.model.mesh.getElementsByType(etype)[0])
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        return {"path": str(out), "volume_elements": counts, "nodes": len(node_tags),
                "groups": {k: len(v) for k, v in groups.items()}}
    finally:
        gmsh.finalize()
