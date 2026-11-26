# Learning notes: flame-to-flow

## 1. The idea

One engineering result is passed through three tools, and every hand-off is checked against an answer known without the tools:

```text
Cantera (GRI-Mech 3.0)                      OpenFOAM (simpleFoam)                      checked against
  flame temperature, flame speed,   nu, rho     channel on a blockMesh mesh   ----->  Poiseuille: u_max/U = 1.5,
  ignition delay, burnt-gas         ------->    channel on a Gmsh mesh                 dp/dx = -12 mu U / h^2,
  viscosity and density at 1500 K               (gmshToFoam)                           entrance length (Durst 2005)
  checked against: 2225 K, heat-flux            sampled with postProcess               blockMesh vs Gmsh within 1%
  measurements (Hermanns 2007)
```

Two tools agreeing with each other and with theory is the evidence that each was used correctly. When a check fails, the question is always "which assumption is wrong: the mesh, the convergence, the boundary conditions, or the units?", never "how much should I loosen the tolerance?".

## 2. Cantera

**What it is.** A C++ library with a Python interface for chemical kinetics, thermodynamics and transport. You load a mechanism (a YAML file listing species, their thermodynamic fits, reactions with rate constants, and transport data) into a `Solution`, set a state, and ask it for properties, equilibrium, or the solution of reactor and 1-D flame problems. GRI-Mech 3.0 (`gri30.yaml`, 53 species, 325 reactions) is the classic natural-gas mechanism.

**The concepts that matter.**
1. **A `Solution` is thermo + kinetics + transport for one gas.** `gas.TP = 300, ct.one_atm` sets the state; `gas.set_equivalence_ratio(phi, "CH4", "O2:1, N2:3.76")` sets the composition. Everything else (`gas.density`, `gas.viscosity`, `gas.net_production_rates`) is computed from that state.
2. **Equilibrium is a constrained minimisation.** `gas.equilibrate("HP")` keeps enthalpy and pressure fixed: that is an adiabatic, constant-pressure burn, so the final temperature is the adiabatic flame temperature. Dissociation (CO, H2, OH, O, H at 2200 K) is why it peaks slightly rich and is lower than a "complete combustion" estimate.
3. **A freely propagating flame is an eigenvalue problem.** `FreeFlame` solves the steady 1-D species and energy equations on an adaptive grid; the inlet velocity that makes the flame stand still is the laminar flame speed S_L. The grid is refined until no solution component changes by more than `slope` (fraction of its range) between points, nor its slope by more than `curve`, with neighbouring intervals within `ratio`. Convection is upwinded, so the error falls roughly in proportion to the spacing (first order): refining from 125 to 1,907 points moved S_L from 0.385 to 0.374 m/s.
4. **Reactors integrate in time.** `IdealGasConstPressureReactor` in a `ReactorNet` is a closed, adiabatic, constant-pressure mixture; `net.step()` takes one adaptive (stiff) integrator step. Ignition delay is when the temperature rises fastest. Its logarithm is nearly linear in 1/T0: an apparent activation energy.
5. **Transport properties depend on composition and temperature.** `gas.viscosity` uses the mixture-averaged model by default. The burnt gas is mostly N2, H2O and CO2, and its viscosity rises with temperature while its density falls, so the kinematic viscosity nu = mu/rho grows fast: at 1500 K it is 16 times that of room-temperature air.

**Where it is used here.** `f2f/flame.py` (all four calculations), `scripts/flame_study.py` (the full sweep and plots), `tests/test_flame.py`.

**Pitfalls met.**
- The first flame solve takes 35-45 s on this laptop (Cantera builds the solution in stages from a guess); a sweep that starts each flame from a converged neighbour takes 5-15 s per flame.
- Cantera's default of 1,000 grid points stopped the finest grid study level (`max number of grid points reached`); `set_max_grid_points` raises it.
- Continuing a sweep from the previous phi makes the grid depend on the order of the sweep (points are only added). Here every phi starts from the same stoichiometric solution.
- The measurements are at 298 K and the requested sweep at 300 K: 2 K are worth about 1% of S_L, the same size as the grid error at the sweep's settings.
- Cantera 3.2 warns that reactors will copy their `Solution` by default after 3.2 (`clone=`); the code passes `clone=True` and reads the reactor's own state.

## 3. OpenFOAM

**What it is.** An open-source C++ finite-volume toolbox: a set of solvers (applications) and utilities that read and write a case directory of text dictionaries. This repo uses OpenFOAM v2406 from openfoam.com (ESI-OpenCFD), run headless in the `opencfd/openfoam-run:2406` Docker image in CI.

**The concepts that matter.**
1. **A case is a directory.** `0/` holds the initial and boundary conditions per field (`U`, `p`), `constant/` the mesh (`polyMesh/`) and physical properties (`transportProperties`, `turbulenceProperties`), `system/` the run control (`controlDict`), discretisation (`fvSchemes`), solvers and algorithm settings (`fvSolution`), mesh input (`blockMeshDict`) and function objects (`sample`). Every application reads the case it is run in.
2. **The mesh is cells, faces and patches.** Boundary faces are grouped into patches, and each patch has a geometric type: `patch` (generic), `wall`, `empty` (the two faces of a one-cell-thick 2-D mesh, where nothing is solved), `symmetryPlane`, and so on. Field boundary conditions (`fixedValue`, `zeroGradient`, `noSlip`, `empty`) are set per patch in `0/`, and they must agree with the geometric type: an `empty` condition on a patch that is not of type `empty` is an error.
3. **simpleFoam solves steady incompressible flow with SIMPLE.** Each iteration solves momentum with the old pressure, then a pressure equation that makes the velocity satisfy continuity, then corrects. `consistent yes` (SIMPLEC) allows light under-relaxation (0.9 here). `residualControl` stops when every field's initial residual is below its tolerance. The run is converged when the result no longer changes, not merely when it stopped.
4. **Incompressible pressure is kinematic.** `p` in incompressible solvers is pressure divided by density, in m^2/s^2. The pressure gradient in Pa/m is rho times the gradient of `p`; `nu` in `transportProperties` is the only property the solver sees.
5. **Discretisation sets the error you can expect.** `Gauss linear` gradients and laplacian, and `linearUpwind` convection, are second order on a uniform mesh, so errors should fall 4 times when the cells halve. Near a wall the gradient is taken over half a cell. Solving that 1-D problem by hand (`f2f/channel.py: discrete_poiseuille`) predicts the developed-flow error exactly: dp/dx is low by the factor 1/(1 + 2/ny^2).

**Where it is used here.** `cases/channel/` (templates), `f2f/foam.py` (writing the case, running OpenFOAM natively or through Docker, fixing the boundary file, reading logs and samples), `f2f/study.py` (the four runs and every check), `tests/test_openfoam.py`.

**Pitfalls met.** See section 6: a sample-file layout, a missing cell, a fit window in developing flow.

## 4. Gmsh

**What it is.** A finite-element mesh generator with a built-in geometry kernel and OpenCASCADE, driven from its GUI, `.geo` scripts or an API (C++, C, Python, Julia). Here the Python API (`pip install gmsh`) makes the same channel mesh as blockMesh, independently.

**The concepts that matter.**
1. **Entities and dimTags.** Geometry is points (dim 0), curves (1), surfaces (2) and volumes (3); the API refers to them as `(dim, tag)`. With the built-in kernel (`gmsh.model.geo`) nothing exists in the model until `synchronize()`.
2. **Transfinite meshing gives structured grids.** `setTransfiniteCurve(c, n)` fixes n nodes on a curve (optionally with a `"Progression"` or `"Bump"` distribution), `setTransfiniteSurface(s)` maps a structured grid between four corners, and `setRecombine(2, s)` pairs the triangles into quadrilaterals.
3. **Extrusion with layers.** `extrude([(2, s)], 0, 0, dz, numElements=[1], recombine=True)` sweeps the quads one layer into hexahedra (without `recombine` they become prisms) and returns the new top surface, volume and side surfaces.
4. **Physical groups decide what is saved and what it is called.** Once any physical group exists, only elements in physical groups are written. `gmshToFoam` turns 2-D physical groups into patches named after the groups and 3-D ones into cell zones.
5. **File formats.** `Mesh.MshFileVersion 2.2` with `Mesh.Binary 0` writes the old ASCII format that `gmshToFoam` reads; the default is 4.1.

**Where it is used here.** `f2f/gmsh_channel.py`, `tests/test_gmsh.py` (counts, groups and spacing, read back with meshio), and the `gmsh_medium` case in `f2f/study.py`.

**Pitfalls met.**
- `gmshToFoam` makes every patch type `patch`. `frontAndBack` must become `empty` and `walls` `wall` before the solver runs (`foam.fix_boundary` edits `constant/polyMesh/boundary`; `foamDictionary` or `changeDictionary` would also do).
- The extruded side surfaces come back in the order of the boundary curves; the code classifies them by bounding box instead, so a change in the curve order can't swap the inlet and a wall.
- The pip `gmsh` wheel links against X11 and GLU libraries even when used headless, so CI installs `libglu1-mesa libxcursor1 libxft2 libxinerama1` first (done up front, never tried without).
- Same cells, same answer: the medium Gmsh mesh and the medium blockMesh mesh gave identical results to 1e-9 %. That proves the workflow, not mesh independence.

## 5. How the tools hand over

| From | To | What is passed | How it is checked |
|---|---|---|---|
| Cantera `product_properties` | `constant/transportProperties` | nu = mu/rho of the burnt gas at 1500 K: 2.414e-4 m^2/s | ideal-gas density check; nu vs frozen composition (0.8%) |
| Cantera | `0/U` | U = Re nu / h = 4.8285 m/s for Re = 200 | Re recomputed in the report |
| Cantera | the checks | rho = 0.2245 kg/m^3, to turn kinematic p into Pa | dp/dx in Pa = rho times the slope of p |
| `f2f/channel.py` | `system/blockMeshDict`, `gmsh_channel.py` | h, length, thickness, nx, ny | checkMesh cell counts; meshio node spacing |
| Gmsh `.msh` 2.2 | `gmshToFoam` | hexahedra and named boundary quads | patch face counts in checkMesh (walls: 480 faces) |
| OpenFOAM `postProcess` | Python | raw `.xy` set files | one sample per cell; their mean equals the inlet velocity |

## 6. What broke while building

1. The first CI run crashed in the analysis: OpenFOAM v2406 writes all the sampled fields of a set into one file (`centreline_p_U.xy`), and my reader split the columns as if both fields had the same width.
2. The mass check failed by 7.8% on the coarse mesh: a `midPoint` line that starts inside the first cell gets no sample for it (9 samples for 10 cells). Lines now start and end outside the channel.
3. dp/dx was 0.1% off the discrete answer on every mesh, so the observed order jumped to 3.8. The fit window began at 15 h, where the pressure gradient was still developing (0.6% steeper). The centreline velocity reaches 99% by 9 h; the pressure gradient needs about twice that.
4. The flame speed moved from 0.385 to 0.374 m/s as the grid went from 125 to 1,907 points (first order). Always quote the refinement settings with a flame speed.
5. Cantera stopped at its default of 1,000 grid points on the finest level.
6. Two of my hand-calculated test expectations (Durst's L/h at Re = 200, the 1-D profile error constant) were slightly wrong; the tests failed and I recomputed them instead of loosening the tolerance.

## 7. Explain it without notes

1. What does `gas.equilibrate("HP")` hold constant, and why does that give the adiabatic flame temperature?
2. Why does the adiabatic flame temperature peak slightly rich of stoichiometric?
3. What problem does `FreeFlame` solve, and why is the flame speed an eigenvalue of it?
4. What do `ratio`, `slope` and `curve` control? Why did the flame speed keep falling as they were tightened, and how would you report a number you trust?
5. Why does the computed flame speed differ from Hermanns' measurement by 2.3% at stoichiometric but 16% at phi = 0.7?
6. How is ignition delay defined here, and what does the slope of ln(tau) against 1/T0 tell you?
7. What are `0/`, `constant/` and `system/` for in an OpenFOAM case? Which file sets nu, which the inlet velocity, which the schemes?
8. What is an `empty` patch, and what goes wrong if `frontAndBack` stays a `patch` after `gmshToFoam`?
9. OpenFOAM reports p = 48.4 at the inlet of the medium case. In what units, and how do you get pascals?
10. How does SIMPLE(C) work, what does `residualControl` stop on, and how did you check the runs were converged?
11. Why is dp/dx on 20 cells 0.5% low, and how could you predict that number before running OpenFOAM?
12. What do transfinite curves, recombination and extrusion with `numElements=[1], recombine=True` each do in Gmsh?
13. Why must the Gmsh mesh have physical groups, and how do their names reach OpenFOAM?
14. The blockMesh and Gmsh results agree to 1e-9 %. What does that prove, and what does it not?
15. Why did the pressure-gradient fit need to start further downstream than the entrance length?

## 8. Your turn

One exercise per tool, 20-40 minutes each. Write the code yourself; the commands below are only the frame. Each was run once while building (Cantera locally, OpenFOAM in CI on a throwaway branch), so the expected results are real numbers.

### Cantera: flame speed at 5 atm (local, about 25 minutes)

1. Make a file `try_pressure.py` outside `f2f/` and write a stoichiometric methane-air `FreeFlame` yourself, along these lines:
   ```python
   import cantera as ct, numpy as np
   gas = ct.Solution("gri30.yaml")
   gas.TP = 300, 1 * ct.one_atm            # change 1 to 5 for the second run
   gas.set_equivalence_ratio(1.0, "CH4", "O2:1, N2:3.76")
   f = ct.FreeFlame(gas, width=0.03)
   f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
   f.solve(loglevel=1, auto=True)
   thickness = (f.T[-1] - f.T[0]) / np.gradient(f.T, f.grid).max()
   print(f.velocity[0], len(f.grid), thickness * 1e3, "mm")
   ```
2. Run it at 1 atm, then at 5 atm: `D:/cache/venv-sim/Scripts/python try_pressure.py` (30-70 s each).
3. **Expect:** 1 atm: S_L = 0.381 m/s on 196 points, thickness 0.44 mm. 5 atm: S_L = **0.191 m/s** on 186 points, thickness **0.15 mm**. The speed halves (S_L ~ P^-0.43 between 1 and 5 atm) and the flame gets three times thinner.
4. Explain: why does pressure slow a methane flame (look at the three-body reaction H + O2 + M -> HO2 + M, which competes with chain-branching H + O2 -> O + OH)? Why does the thinner flame need the same number of points but a smaller spacing? Run 5 atm with `width=0.01`: the answer should not change (0.1915 m/s), which tells you the domain was wide enough.

### Gmsh: cluster the cells at the walls (local mesh, CI run, about 30 minutes)

1. Build the graded mesh and look at it in the Gmsh GUI (the pip wheel includes it):
   ```bash
   D:/cache/venv-sim/Scripts/python -c "from f2f import gmsh_channel, channel; print(gmsh_channel.build(channel.Channel(), channel.LEVELS['medium'], 'bump.msh', wall_bump=0.2))"
   D:/cache/venv-sim/Scripts/gmsh bump.msh
   ```
   Check that the cells get thinner towards both walls (zoom in near x = 0).
2. Change it yourself: in `f2f/gmsh_channel.py` replace `"Bump", wall_bump` with `"Progression", 1.2` on the two vertical lines, rebuild, and open it again. The small cells go to the bottom on the right line (drawn upwards, p2 to p3) and to the top on the left line (drawn downwards, p4 to p1), so the transfinite surface joins mismatched spacings and the cells are skewed. Fix it by giving one line `-1.2` (or by reversing that curve), and explain why curve direction matters.
3. Predict before running: `f2f.channel.discrete_poiseuille_faces(yf)`, with `yf` the unique y coordinates of the Bump mesh (`meshio.read('bump.msh').points[:, 1]`), gives what OpenFOAM should get across the channel.
4. Run it in OpenFOAM through CI: `git checkout -b exercise-gmsh`, put `--mesher gmsh --bump 0.2` in `exercise.txt` and a second line `--mesher gmsh` for the uniform reference, commit, `git push -u origin exercise-gmsh`, then `gh run watch` and `gh run view --log | grep -A 12 "case exercise"`. Branches named `exercise-...` skip the Cantera job.
5. **Expect:** wall cells 0.018 h and centre cells 0.080 h instead of 0.05 h everywhere; dp/dx error **-0.75%** against -0.48% on the uniform mesh; profile L2 0.198% against 0.233%; entrance length 9.15 h. The 1-D prediction is -0.77% and 0.198%. Clustering cells at the walls made the pressure gradient *worse*: a parabola has the same curvature everywhere, so there is no thin wall layer to resolve, and the bigger centre cells and uneven spacing cost accuracy. Grading pays off when the solution has a thin layer (a turbulent boundary layer), not here.
6. Delete the branch afterwards: `git push origin --delete exercise-gmsh`.

### OpenFOAM: double the inlet velocity (CI run, about 30 minutes)

1. Read the case first, locally: `D:/cache/venv-sim/Scripts/python scripts/exercise.py --re 400` writes `work/exercise_blockMesh_re400_L30_240x20/` without running it (no OpenFOAM here). Open `0/U` (inlet `fixedValue` 9.657 m/s instead of 4.829), `constant/transportProperties` (nu unchanged), `system/fvSolution` and `system/sample`.
2. Predict: Durst gives L_e = 17.73 h at Re = 400 (8.92 h at 200). The 30 h channel is then too short: the gradient fit at 22-28 h would sit in developing flow, as in section 6, item 3. Double the length too, keeping the cell size: `--re 400 --length-h 60 --nx 480`.
3. Run it in CI: branch `exercise-re400`, `exercise.txt` containing `--re 400 --length-h 60 --nx 480`, push, `gh run watch`, read the log as above. Download the artifact (`gh run download <run id> -n flow-ci-results`) and open `logs/exercise_.../log.simpleFoam` to watch the residuals fall, and `log.checkMesh` for the cell count and mesh quality.
4. **Expect:** converged in 184 iterations; L_e = **18.05 h** (Durst 17.73 h, +1.8%), twice the 9.13 h at Re = 200; dp/dx = -62.50 Pa/m, twice the Re = 200 value, with the same -0.49% error (the error depends on the cells across, not on Re); u_max/U = 1.4963. The local pressure gradient is still 1.4% steeper than developed at 25 h, which is why the 30 h channel would not do.
5. Then run `--re 400` on the 30 h channel and explain the dp/dx error you get.
