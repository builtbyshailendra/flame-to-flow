"""The OpenFOAM side that can be checked without OpenFOAM: the generated case files, the
boundary-type fix after gmshToFoam, and the readers for sampled sets and solver logs."""
import numpy as np
import pytest

from f2f import channel, foam

BOUNDARY_FROM_GMSHTOFOAM = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    location    "constant/polyMesh";
    object      boundary;
}

4
(
    inlet
    {
        type            patch;
        physicalType    patch;
        nFaces          20;
        startFace       9460;
    }
    outlet
    {
        type            patch;
        physicalType    patch;
        nFaces          20;
        startFace       9480;
    }
    walls
    {
        type            patch;
        physicalType    patch;
        nFaces          480;
        startFace       9500;
    }
    frontAndBack
    {
        type            patch;
        physicalType    patch;
        nFaces          9600;
        startFace       9980;
    }
)
"""


def test_case_files_are_filled_in(tmp_path):
    ch = channel.Channel(nu=2.4143e-4, rho=0.22449, re=200)
    case = foam.write_case(tmp_path / "c", ch, channel.LEVELS["coarse"],
                           props={"mu": 5.4198e-5, "rho": 0.22449, "T": 1500})
    for p in case.rglob("*"):
        if p.is_file():
            assert "{{" not in p.read_text(), p
    assert "nu              [0 2 -1 0 0 0 0] 0.00024143;" in (case / "constant/transportProperties").read_text()
    u = (case / "0/U").read_text()
    assert f"uniform ({ch.u_mean:.10g} 0 0)" in u
    bm = (case / "system/blockMeshDict").read_text()
    assert "(120 10 1)" in bm and "(0.3  0.01  0.001)" in bm
    sample = (case / "system/sample").read_text()
    for s in foam.stations(ch):
        assert foam.station_name(s) in sample
    assert foam.station_name(25.03) in sample        # the developed-flow station, 5 h before the outlet
    assert "start   (0.2503 -0.001 0.0005);" in sample  # lines start outside the channel


def test_stations_never_sit_on_a_cell_face():
    for mesh in channel.LEVELS.values():
        dx_h = channel.Channel().length_h / mesh.nx
        for s in foam.stations(channel.Channel()):
            frac = (s / dx_h) % 1.0
            assert 0.05 < frac < 0.95, (mesh, s)


def test_fix_boundary_sets_types(tmp_path):
    b = tmp_path / "boundary"
    b.write_text(BOUNDARY_FROM_GMSHTOFOAM)
    before = foam.fix_boundary(b, {"walls": "wall", "frontAndBack": "empty", "inlet": "patch",
                                   "outlet": "patch"})
    assert before == {"walls": "patch", "frontAndBack": "patch", "inlet": "patch", "outlet": "patch"}
    text = b.read_text()
    assert "physicalType" not in text
    block = text[text.index("walls"):text.index("frontAndBack")]
    assert "type            wall;" in block and "nFaces          480;" in block
    block = text[text.index("frontAndBack"):]
    assert "type            empty;" in block and "startFace       9980;" in block
    with pytest.raises(KeyError):
        foam.fix_boundary(b, {"symmetry": "symmetryPlane"})


def test_read_set_raw_files(tmp_path):
    d = tmp_path / "postProcessing" / "sample" / "1234"
    d.mkdir(parents=True)
    y = np.linspace(0.0005, 0.0095, 10)
    rows = "\n".join(f"0.25 {v:.6g} 0.0005 {6 * v / 0.01 * (1 - v / 0.01):.6g} 0 0" for v in y)
    (d / "x25p03_U.xy").write_text("# x y z U_x U_y U_z\n" + rows + "\n")
    (d / "x25p03_p.xy").write_text("\n".join(f"0.25 {v:.6g} 0.0005 7.5" for v in y) + "\n")
    xyz, u = foam.read_set(tmp_path, "sample", "x25p03", "U")
    assert xyz.shape == (10, 3) and u.shape == (10, 3)
    assert u[:, 0].max() < 1.5
    _, p = foam.read_set(tmp_path, "sample", "x25p03", "p")
    assert p.shape == (10,) and p[0] == 7.5


def test_read_set_one_file_with_all_fields(tmp_path):
    # what OpenFOAM v2406 writes: centreline_p_U.xy with columns x y z p U_x U_y U_z
    d = tmp_path / "postProcessing" / "sampleCentreline" / "812"
    d.mkdir(parents=True)
    (d / "centreline_p_U.xy").write_text("0 0.005 0.0005 47.66 4.83 0.0 0.0\n"
                                         "0.3 0.005 0.0005 0.0 7.10 1e-9 0.0\n")
    xyz, u = foam.read_set(tmp_path, "sampleCentreline", "centreline", "U")
    _, p = foam.read_set(tmp_path, "sampleCentreline", "centreline", "p")
    assert p.tolist() == [47.66, 0.0]
    assert u[:, 0].tolist() == [4.83, 7.10] and xyz[1, 0] == 0.3


def test_read_solver_log(tmp_path):
    log = tmp_path / "log.simpleFoam"
    log.write_text(
        "Time = 1\n\nsmoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.005, No Iterations 3\n"
        "GAMG:  Solving for p, Initial residual = 1, Final residual = 0.009, No Iterations 12\n"
        "ExecutionTime = 0.05 s  ClockTime = 0 s\n\n"
        "Time = 812\n\nsmoothSolver:  Solving for Ux, Initial residual = 9.8e-08, Final residual = 1e-09, No Iterations 1\n"
        "smoothSolver:  Solving for Uy, Initial residual = 3.1e-07, Final residual = 2e-09, No Iterations 1\n"
        "GAMG:  Solving for p, Initial residual = 9.9e-07, Final residual = 9e-09, No Iterations 4\n"
        "ExecutionTime = 4.21 s  ClockTime = 5 s\n\n"
        "SIMPLE solution converged in 812 iterations\n\nEnd\n")
    r = foam.read_solver_log(log)
    assert r["iterations"] == 812 and r["converged"]
    assert r["final_initial_residuals"] == {"Ux": 9.8e-08, "Uy": 3.1e-07, "p": 9.9e-07}
    assert r["execution_s"] == 4.21
