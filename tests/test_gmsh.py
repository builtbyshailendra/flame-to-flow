"""The Gmsh mesh, checked without OpenFOAM: element types and counts, groups, file format."""
import meshio
import numpy as np
import pytest

from f2f import channel, gmsh_channel


@pytest.fixture(scope="module")
def medium(tmp_path_factory):
    out = tmp_path_factory.mktemp("gmsh") / "channel.msh"
    info = gmsh_channel.build(channel.Channel(), channel.LEVELS["medium"], out)
    return out, info


def test_structured_hexes_one_layer(medium):
    path, info = medium
    m = channel.LEVELS["medium"]
    assert info["volume_elements"] == {"Hexahedron 8": m.nx * m.ny}
    assert info["nodes"] == (m.nx + 1) * (m.ny + 1) * 2


def test_msh_22_ascii_with_named_groups(medium):
    path, _ = medium
    head = path.read_text().splitlines()[:12]
    assert head[0] == "$MeshFormat" and head[1].startswith("2.2 0 ")   # version 2.2, ASCII
    text = "\n".join(head)
    for name in ("inlet", "outlet", "walls", "frontAndBack", "fluid"):
        assert f'"{name}"' in text


def test_meshio_reads_the_same_mesh(medium):
    path, _ = medium
    ch, m = channel.Channel(), channel.LEVELS["medium"]
    mesh = meshio.read(path)
    cells = {c.type: len(c.data) for c in mesh.cells}
    assert cells["hexahedron"] == m.nx * m.ny
    # boundary quads: inlet + outlet + 2 walls + front and back
    assert cells["quad"] == 2 * m.ny + 2 * m.nx + 2 * m.nx * m.ny
    assert mesh.points.min(axis=0) == pytest.approx([0, 0, 0])
    assert mesh.points.max(axis=0) == pytest.approx([ch.length, ch.h, ch.dz])
    # uniform spacing across the channel, same as blockMesh
    ys = np.unique(np.round(mesh.points[:, 1] / ch.h, 9))
    assert len(ys) == m.ny + 1
    assert np.diff(ys) == pytest.approx(np.full(m.ny, 1 / m.ny))


def test_bump_clusters_cells_at_the_walls(tmp_path):
    ch, m = channel.Channel(), channel.Mesh(40, 20)
    out = tmp_path / "bump.msh"
    gmsh_channel.build(ch, m, out, wall_bump=0.2)
    ys = np.unique(np.round(meshio.read(out).points[:, 1] / ch.h, 9))
    dy = np.diff(ys)
    assert len(dy) == m.ny
    assert dy[0] < 0.5 / m.ny and dy[-1] < 0.5 / m.ny     # wall cells under half the uniform size
    assert dy[m.ny // 2 - 1] > 1 / m.ny                   # centre cells bigger
