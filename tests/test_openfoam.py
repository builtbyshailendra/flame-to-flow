"""OpenFOAM against Poiseuille, on three blockMesh levels and one Gmsh mesh.

Runs the four cases once per session (about 2-4 minutes); skipped when OpenFOAM is missing
(locally on Windows). CI runs it with the opencfd/openfoam-run:2406 Docker image.
"""
import pytest

from f2f import foam, study
from f2f.channel import LEVELS, observed_order

pytestmark = [
    pytest.mark.foam,
    pytest.mark.skipif(foam.foam_mode() is None,
                       reason="OpenFOAM not available (no simpleFoam on PATH, F2F_FOAM_IMAGE unset)"),
]


@pytest.fixture(scope="session")
def result():
    return study.run_study()


@pytest.fixture(scope="session")
def cases(result):
    return result["cases"]


BLOCK = [f"blockMesh_{k}" for k in LEVELS]
ALL = BLOCK + ["gmsh_medium"]


@pytest.mark.parametrize("name", ALL)
def test_mesh_is_valid_and_has_the_planned_cells(cases, name):
    c = cases[name]
    assert c["checkMesh"]["ok"], c["checkMesh"]
    assert c["checkMesh"]["cells"] == c["mesh"]["cells"]


@pytest.mark.parametrize("name", ALL)
def test_solver_converged(cases, name):
    s = cases[name]["solver"]
    assert s["converged"], s
    assert max(s["final_initial_residuals"].values()) < 1e-5


@pytest.mark.parametrize("name", ALL)
def test_mass_is_conserved(result, name):
    # cell values on uniform cells: their mean is the flow rate over the height
    p = result["cases"][name]["profile"]
    assert p["u_mean_sampled"] == pytest.approx(result["channel"]["u_mean"], rel=1e-4)


@pytest.mark.parametrize("name,tol", [("blockMesh_coarse", 0.015), ("blockMesh_medium", 0.005),
                                      ("blockMesh_fine", 0.002), ("gmsh_medium", 0.005)])
def test_peak_over_mean_is_one_and_a_half(cases, name, tol):
    assert cases[name]["profile"]["umax_ratio"] == pytest.approx(1.5, rel=tol)


@pytest.mark.parametrize("name,tol", [("blockMesh_coarse", 0.015), ("blockMesh_medium", 0.005),
                                      ("blockMesh_fine", 0.002), ("gmsh_medium", 0.005)])
def test_profile_matches_poiseuille(cases, name, tol):
    p = cases[name]["profile"]
    assert p["l2_rel"] < tol
    assert p["v_max_rel"] < 1e-3            # no cross-flow left where the flow is developed


@pytest.mark.parametrize("name,tol", [("blockMesh_coarse", 0.025), ("blockMesh_medium", 0.01),
                                      ("blockMesh_fine", 0.003), ("gmsh_medium", 0.01)])
def test_pressure_gradient_matches_12_mu_U_over_h2(result, name, tol):
    p = result["cases"][name]["pressure"]
    assert p["rel_error"] == pytest.approx(0, abs=tol)
    # OpenFOAM's p is p/rho: Pa = rho * kinematic, and the exact value is -12 mu U / h^2
    ch = result["channel"]
    mu = ch["nu"] * ch["rho"]
    assert p["dpdx_pa_exact"] == pytest.approx(-12 * mu * ch["u_mean"] / ch["h"] ** 2, rel=1e-12)
    assert p["dpdx_pa"] == pytest.approx(p["dpdx_kinematic"] * ch["rho"], rel=1e-12)


@pytest.mark.parametrize("name", ALL)
def test_solution_is_the_discrete_one(cases, name):
    # On ny uniform cells the developed solution should be the 1-D discrete solution
    # (dp/dx = exact / (1 + 2/ny^2)), not the exact one: the remaining error is the mesh's.
    p = cases[name]["pressure"]
    assert p["rel_error_vs_discrete_1d"] == pytest.approx(0, abs=1e-3)


def test_errors_shrink_at_second_order(cases):
    for key, sub in (("pressure", "rel_error"), ("profile", "l2_rel")):
        errs = [abs(cases[n][key][sub]) for n in BLOCK]
        orders = observed_order(errs)
        assert all(1.7 < o < 2.3 for o in orders), (key, errs, orders)


@pytest.mark.parametrize("name", ["blockMesh_medium", "blockMesh_fine", "gmsh_medium"])
def test_entrance_length_matches_durst(cases, name):
    e = cases[name]["entrance"]
    assert e["L_e_over_h"] == pytest.approx(e["durst_over_h"], rel=0.10)


def test_gmsh_and_blockmesh_agree_within_one_percent(cases):
    b, g = cases["blockMesh_medium"], cases["gmsh_medium"]
    assert g["pressure"]["dpdx_kinematic"] == pytest.approx(b["pressure"]["dpdx_kinematic"], rel=0.01)
    assert g["profile"]["umax_ratio"] == pytest.approx(b["profile"]["umax_ratio"], rel=0.01)
    assert g["entrance"]["L_e_m"] == pytest.approx(b["entrance"]["L_e_m"], rel=0.01)
    assert g["pressure"]["p_inlet_kinematic"] == pytest.approx(b["pressure"]["p_inlet_kinematic"], rel=0.01)


def test_gmshtofoam_patch_types_needed_fixing(cases):
    before = cases["gmsh_medium"]["gmsh"]["patch_types_from_gmshToFoam"]
    assert before["frontAndBack"] != "empty" and before["walls"] != "wall"
    assert cases["gmsh_medium"]["checkMesh"]["patch_faces"].get("walls") == 2 * LEVELS["medium"].nx
