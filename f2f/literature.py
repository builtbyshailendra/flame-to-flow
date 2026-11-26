"""Known answers the computed results are checked against.

Every number here comes from a cited source; nothing is fitted to this project's results.
"""

# Laminar burning velocity of methane-air, 1 atm, unburnt gas at 298 K, measured with the
# heat flux method (flat adiabatic flame, no stretch correction needed).
# Source: R.T.E. Hermanns, "Laminar burning velocities of methane-hydrogen-air mixtures",
# PhD thesis, Eindhoven University of Technology, 2007, Appendix E, Table E.3 (R_H2 = 0%).
# doi:10.6100/IR630126. Values in cm/s as printed: (phi, S_L, uncertainty).
HERMANNS_2007_SOURCE = (
    "Hermanns, R.T.E. (2007). Laminar burning velocities of methane-hydrogen-air mixtures. "
    "PhD thesis, TU Eindhoven, Table E.3 (heat flux method, 1 atm, 298 K). doi:10.6100/IR630126"
)
HERMANNS_2007_T_UNBURNT = 298.0
HERMANNS_2007 = [
    (0.65, 12.0, 0.8),
    (0.70, 16.7, 0.6),
    (0.75, 21.5, 0.4),
    (0.80, 25.8, 0.2),
    (0.85, 29.6, 0.1),
    (0.90, 32.7, 0.2),
    (0.95, 34.9, 0.1),
    (1.00, 36.4, 0.1),
    (1.05, 37.2, 0.1),
    (1.10, 37.0, 0.1),
    (1.15, 35.6, 0.1),
    (1.20, 32.7, 0.1),
    (1.25, 28.3, 0.2),
    (1.30, 23.0, 0.3),
    (1.35, 18.9, 0.2),
    (1.40, 15.6, 0.2),
    (1.45, 13.3, 0.3),
    (1.50, 11.6, 0.2),
]


def hermanns_speed(phi: float) -> float:
    """Measured S_L in m/s at a tabulated phi (exact match within 1e-6)."""
    for p, s, _ in HERMANNS_2007:
        if abs(p - phi) < 1e-6:
            return s / 100.0
    raise KeyError(f"phi={phi} is not in Hermanns' table")


# Development (entrance) length of laminar flow between parallel plates with a uniform inlet
# profile: the distance where the centreline velocity reaches 99% of its fully developed value.
# Durst, Ray, Unsal and Bayoumi, "The development lengths of laminar pipe and channel flows",
# J. Fluids Eng. 127 (2005) 1154-1160. Written with the hydraulic diameter D_h = 2H and
# Re_Dh = U D_h / nu it reads L/D_h = [0.3155^1.6 + (0.01105 Re_Dh)^1.6]^(1/1.6)
# (as quoted in Ray, Durst and Ray, arXiv:1707.04947, Eq. 2b). With the channel height H and
# Re_H = U H / nu that is L/H = [0.631^1.6 + (0.0442 Re_H)^1.6]^(1/1.6).
DURST_2005_SOURCE = (
    "Durst, Ray, Unsal, Bayoumi (2005), J. Fluids Eng. 127:1154-1160, channel correlation "
    "(quoted in Ray, Durst, Ray, arXiv:1707.04947, Eq. 2b)"
)


def durst_entrance_length(re_h: float, h: float) -> float:
    """Entrance length in metres for a channel of height h at Re_h = U_mean h / nu."""
    return h * (0.631 ** 1.6 + (0.0442 * re_h) ** 1.6) ** (1 / 1.6)
