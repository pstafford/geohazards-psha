"""A complete probabilistic seismic hazard analysis (PSHA) in a few hundred lines of numpy.

Geotechnical Hazards (Imperial College London): the course's toy PSHA model.

The model: a site at the origin, two seismic sources
  * a vertical strike-slip FAULT, 115 km long, 50 km north of the site, rupturing the
    seismogenic layer (0-18 km); characteristic magnitude-frequency distribution
    (Youngs & Coppersmith, 1985);
  * an AREA source, a circle of radius 70 km centred on the site; point ruptures at 9 km
    depth; truncated Gutenberg-Richter distribution;
and the Chiou & Youngs (2014) ground-motion model.

Every modelling choice follows the conventions of OpenQuake hazardlib (how ruptures float
on the fault, how the area is gridded, how magnitudes are binned, how the ground-motion
distribution is truncated), so the hazard computed here can be checked against OpenQuake.

The calculation has three steps, all in plain numpy:
  1. list every rupture scenario once: magnitude, annual rate and geometry (distances);
  2. for each scenario, compute the median and standard deviation of ln(ground motion);
  3. the hazard curve is the sum, over scenarios, of  rate x P(IM > im | scenario).

    lambda(IM > im) = sum_i  P(IM > im | rup_i)  lambda(rup_i)

Units: km, g, years. Magnitudes are moment magnitudes, M0 = 10^(1.5 M + 9.05) N m.
"""
from pathlib import Path

import numpy as np
from scipy.special import ndtr, expit   # standard normal CDF Phi(x); logistic 1/(1+e^-x)

HERE = Path(__file__).resolve().parent

# ======================================================================= model parameters
# Change these (or pass a modified copy to the functions below) to explore sensitivities.
PARAMS = dict(
    # site
    vs30=760.0,                 # m/s, measured
    # fault source: trace along y = fault_dist, from x = -fault_left to x = +fault_right
    fault_dist=50.0, fault_left=35.0, fault_right=80.0,
    fault_top=0.0, fault_bottom=18.0,       # seismogenic depths (km); vertical, strike-slip
    fault_mmin=5.0, fault_b=1.2, fault_mchar=7.45,   # characteristic box 7.2-7.7
    fault_slip_rate=30.0,       # mm/yr: sets the fault's activity by moment balance
    shear_modulus=3.3e10,       # N/m^2
    # area source: circle of radius area_radius about the site
    area_radius=70.0, area_depth=9.0,
    area_mmin=5.0, area_mmax=6.5, area_b=1.0,
    area_rate=0.5,              # annual rate of events with M >= area_mmin
    # discretisation (OpenQuake: rupture_mesh_spacing, area_source_discretization)
    dm=0.1, mesh=1.0, area_grid=2.0, aspect=1.5,
    # ground-motion distribution truncated at +/- trunc standard deviations
    trunc=3.0,
)


# ======================================================================= magnitudes
def bin_centres(mmin, mmax, dm):
    """Magnitude bin centres from mmin + dm/2 to mmax - dm/2 (OpenQuake's convention)."""
    lo, hi = round(mmin / dm) * dm, round(mmax / dm) * dm
    n = int(round((hi - lo) / dm))
    return lo + dm * (np.arange(n) + 0.5)


def gr_rates(mmin, mmax, b, rate, dm):
    """Truncated Gutenberg-Richter: annual rate in each magnitude bin.

    N(M >= m) = rate * (10^(-b m) - 10^(-b mmax)) / (10^(-b mmin) - 10^(-b mmax))
    """
    m = bin_centres(mmin, mmax, dm)
    N = lambda x: rate * (10 ** (-b * x) - 10 ** (-b * mmax)) / (10 ** (-b * mmin) - 10 ** (-b * mmax))
    return m, N(m - dm / 2) - N(m + dm / 2)


def yc85_rates(mmin, b, mchar, rate, dm):
    """Characteristic model of Youngs & Coppersmith (1985), as in OpenQuake.

    Exponential (Gutenberg-Richter) below mchar - 0.25; a uniform 'box' of characteristic
    events between mchar - 0.25 and mchar + 0.25 whose rate density equals the exponential
    rate density one magnitude unit below the box. Scaled so that N(M >= mmin) = rate.
    """
    m = bin_centres(mmin, mchar + 0.25, dm)
    # unscaled: incremental rate density of the exponential part is 10^(-b m) b ln10
    a_unit = 0.0
    def n_exp(lo, hi):                       # rate of exponential events in [lo, hi)
        return 10 ** (a_unit - b * lo) - 10 ** (a_unit - b * hi)
    box_density = b * np.log(10) * 10 ** (a_unit - b * (mchar - 1.25))   # per unit magnitude
    r = np.where(m < mchar - 0.25, n_exp(m - dm / 2, m + dm / 2), box_density * dm)
    return m, r * rate / r.sum()


def moment(m):
    """Seismic moment (N m) of moment magnitude m."""
    return 10 ** (1.5 * np.asarray(m) + 9.05)


def fault_mfd(p=PARAMS):
    """The fault's magnitude bins and annual rates, from its slip rate (moment balance):
    sum over bins of rate x M0(m) = shear modulus x fault area x slip rate."""
    m, r = yc85_rates(p["fault_mmin"], p["fault_b"], p["fault_mchar"], 1.0, p["dm"])
    area = (p["fault_left"] + p["fault_right"]) * (p["fault_bottom"] - p["fault_top"]) * 1e6   # m^2
    moment_rate = p["shear_modulus"] * area * p["fault_slip_rate"] / 1000.0                  # N m / yr
    return m, r * moment_rate / np.sum(r * moment(m))


# ======================================================================= rupture scenarios
def rupture_area(m):
    """Median rupture area (km^2), Wells & Coppersmith (1994), strike-slip."""
    return 10 ** (-3.42 + 0.90 * m)


def fault_ruptures(p=PARAMS):
    """All ruptures on the fault: OpenQuake's 'floating rupture' algorithm.

    The fault plane is a mesh of points every p['mesh'] km. For each magnitude, a rupture of
    area rupture_area(m) and aspect ratio (length/width) p['aspect'] is placed at every
    possible position on the mesh (shrunk to fit the fault if needed); the magnitude's rate
    is shared equally among the positions.
    """
    s = p["mesh"]
    L = p["fault_left"] + p["fault_right"]
    W = p["fault_bottom"] - p["fault_top"]
    ncols, nrows = int(round(L / s)) + 1, int(round(W / s)) + 1
    xs = -p["fault_left"] + s * np.arange(ncols)            # mesh columns (along strike)
    zs = p["fault_top"] + s * np.arange(nrows)              # mesh rows (depth)
    L, W = (ncols - 1) * s, (nrows - 1) * s
    out = []
    for m, rate in zip(*fault_mfd(p)):
        area = rupture_area(m)
        length = np.sqrt(area * p["aspect"])
        width = area / length
        if area >= L * W:                                   # bigger than the fault
            length, width = L, W
        elif width > W:                                     # too wide: make it longer
            length, width = length * width / W, W
        elif length > L:                                    # too long: make it wider
            length, width = L, width * length / L
        rc, rr = int(round(length / s)) + 1, int(round(width / s)) + 1
        nx, nz = ncols - rc + 1, nrows - rr + 1
        for i in range(nz):                                 # every position down dip ...
            for j in range(nx):                             # ... and along strike
                x0, x1, ztor = xs[j], xs[j + rc - 1], zs[i]
                dx = max(x0, 0.0, -x1)                      # along-strike offset of site from rupture
                rjb = np.hypot(dx, p["fault_dist"])
                out.append((0, m, rate / (nx * nz), rjb, np.hypot(rjb, ztor), ztor, x0, x1))
    return out


def area_points(p=PARAMS):
    """Grid of point-source locations inside the area source (OpenQuake's discretisation:
    a square grid every p['area_grid'] km, starting from the north-west corner)."""
    R, g = p["area_radius"], p["area_grid"]
    c = np.arange(-R, R + 1e-9, g)
    X, Y = np.meshgrid(c, -c)
    inside = np.hypot(X, Y) < R - 1e-6
    return X[inside], Y[inside]


def area_ruptures(p=PARAMS):
    """Point ruptures at depth p['area_depth'] at every grid point, rate shared equally."""
    x, y = area_points(p)
    repi = np.hypot(x, y)
    out = []
    for m, rate in zip(*gr_rates(p["area_mmin"], p["area_mmax"], p["area_b"],
                                 p["area_rate"], p["dm"])):
        r = rate / len(x)
        for re in repi:
            out.append((1, m, r, re, np.hypot(re, p["area_depth"]), p["area_depth"], np.nan, np.nan))
    return out


SCENARIO_FIELDS = ["source", "mag", "rate", "rjb", "rrup", "ztor", "x0", "x1"]


def scenarios(p=PARAMS):
    """Step 1: every rupture scenario, once. Returns a dict of numpy arrays (columns):
    source (0 = fault, 1 = area), mag, rate (per year), rjb, rrup, ztor (km), and the
    along-strike extent x0..x1 of fault ruptures."""
    rows = np.array(fault_ruptures(p) + area_ruptures(p))
    return {k: rows[:, i] for i, k in enumerate(SCENARIO_FIELDS)}


# ======================================================================= ground motion
_COEFFS = np.genfromtxt(HERE / "cy14_coefficients.csv", delimiter=",", names=True, skip_header=1)
PERIODS = _COEFFS["period"]            # 0 = PGA


def cy14(T, mag, rrup, rjb, ztor, vs30=760.0, dip=90.0, rake=0.0):
    """Step 2: Chiou & Youngs (2014): median ln(IM) (g) and total standard deviation.

    Strike-slip (rake 0) and vertical (dip 90) ruptures, so the style-of-faulting, dip and
    hanging-wall terms vanish; measured Vs30; basin depth z1.0 equal to its mean for the Vs30
    (so the basin term vanishes); no directivity. T = 0 gives PGA. Only tabulated periods.
    """
    row = np.flatnonzero(np.isclose(PERIODS, T))
    if not len(row):
        raise ValueError(f"period {T} s is not in the coefficient table: {list(PERIODS)}")
    C = _COEFFS[row[0]]
    mag, rrup, ztor = np.asarray(mag, float), np.asarray(rrup, float), np.asarray(ztor, float)
    coshm = np.cosh(2.0 * np.clip(mag - 4.5, 0.0, None))
    # top-of-rupture depth relative to its average for the magnitude (strike-slip)
    dztor = ztor - np.clip(2.673 - 1.136 * np.clip(mag - 4.970, 0.0, None), 0.0, None) ** 2
    # reference motion on rock (Vs30 = 1130 m/s), CY14 eq. 11
    ln_yref = (C["c1"]
               + (C["c7"] + C["c7b"] / coshm) * dztor
               + (0.0 + C["c11b"] / coshm) * np.cos(np.radians(dip)) ** 2
               + 1.06 * (mag - 6.0) + ((1.06 - C["c3"]) / C["cn"]) * np.log1p(np.exp(C["cn"] * (C["cm"] - mag)))
               - 2.1 * np.log(rrup + C["c5"] * np.cosh(C["c6"] * np.clip(mag - C["chm"], 0.0, None)))
               + (-0.5 + 2.1) * np.log(np.sqrt(rrup ** 2 + 50.0 ** 2))
               + (C["cg1"] + C["cg2"] / np.cosh(np.clip(mag - C["cg3"], 0.0, None))) * rrup)
    yref = np.exp(ln_yref)
    # site amplification, linear and nonlinear (CY14 eq. 12)
    vs = min(vs30, 1130.0)
    f_lin = C["phi1"] * min(np.log(vs30 / 1130.0), 0.0)
    b_nl = C["phi2"] * (np.exp(C["phi3"] * (vs - 360.0)) - np.exp(C["phi3"] * (1130.0 - 360.0)))
    mean = ln_yref + f_lin + b_nl * np.log((yref + C["phi4"]) / C["phi4"])
    # standard deviation (CY14 eq. 13), measured Vs30
    nl0 = b_nl * yref / (yref + C["phi4"])
    mm = np.clip(mag - 5.0, 0.0, 1.5)
    tau = C["tau1"] + (C["tau2"] - C["tau1"]) / 1.5 * mm
    phi = (C["sig1"] + (C["sig2"] - C["sig1"]) / 1.5 * mm) * np.sqrt(0.7 + (1.0 + nl0) ** 2)
    sigma = np.sqrt((1.0 + nl0) ** 2 * tau ** 2 + phi ** 2)
    return mean, sigma


def prob_exceed(im, mean, sigma, trunc=PARAMS["trunc"]):
    """P(IM > im) for a lognormal distribution truncated at +/- trunc standard deviations."""
    eps = (np.log(im) - mean) / sigma
    lo, hi = ndtr(-trunc), ndtr(trunc)
    return np.clip((hi - ndtr(eps)) / (hi - lo), 0.0, 1.0)


# ======================================================================= hazard
def hazard_curve(T, imls, sc=None, p=PARAMS, by_source=False):
    """Step 3: lambda(IM > im) = sum_i P(IM > im | rup_i) lambda(rup_i), for each im in imls.

    With by_source=True, also returns the curve of each source (fault, area)."""
    sc = scenarios(p) if sc is None else sc
    mean, sigma = cy14(T, sc["mag"], sc["rrup"], sc["rjb"], sc["ztor"], p["vs30"])
    poe = prob_exceed(np.asarray(imls, float)[:, None], mean[None, :], sigma[None, :], p["trunc"])
    contrib = poe * sc["rate"][None, :]
    total = contrib.sum(axis=1)
    if not by_source:
        return total
    return total, [contrib[:, sc["source"] == s].sum(axis=1) for s in (0, 1)]


def im_at_rate(T, rate, sc=None, p=PARAMS, imls=None):
    """Intensity with annual exceedance rate `rate` (log-log interpolation of the curve)."""
    imls = np.logspace(-3, 1, 200) if imls is None else np.asarray(imls, float)
    lam = hazard_curve(T, imls, sc, p)
    ok = lam > 0
    return float(np.exp(np.interp(np.log(rate), np.log(lam[ok][::-1]), np.log(imls[ok][::-1]))))


# ======================================================================= disaggregation
def disaggregate(T, im, sc=None, p=PARAMS):
    """Which scenarios cause IM > im? Returns, for every scenario, its contribution
    P(IM > im | rup_i) lambda(rup_i) to the hazard and its epsilon.

    Divided by their sum, lambda(IM > im), the contributions are P(rup_i | IM > im).
    epsilon_i = (ln im - mean_i) / sigma_i is how many standard deviations above its own
    median scenario i must be to reach im."""
    sc = scenarios(p) if sc is None else sc
    mean, sigma = cy14(T, sc["mag"], sc["rrup"], sc["rjb"], sc["ztor"], p["vs30"])
    return prob_exceed(im, mean, sigma, p["trunc"]) * sc["rate"], (np.log(im) - mean) / sigma


# ======================================================================= conditional spectra
def correlation(T1, T2):
    """Correlation of epsilons at periods T1 and T2 (Baker & Jayaram, 2008).
    PGA (T = 0) is treated as T = 0.01 s."""
    t1, t2 = np.maximum(T1, 0.01), np.maximum(T2, 0.01)
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    c1 = 1 - np.cos(np.pi / 2 - 0.366 * np.log(tmax / np.maximum(tmin, 0.109)))
    c2 = np.where(tmax < 0.2, 1 - 0.105 * (1 - expit(5 - 100 * tmax))
                  * (tmax - tmin) / (tmax - 0.0099), 0.0)
    c3 = np.where(tmax < 0.109, c2, c1)
    c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * (1 + np.cos(np.pi * tmin / 0.109))
    return np.where(tmax < 0.109, c2, np.where(tmin > 0.109, c1, np.where(tmax < 0.2, np.minimum(c2, c4), c4)))


def conditional_spectrum(Tstar, im, sc=None, p=PARAMS, periods=PERIODS, sources=(0, 1)):
    """Conditional mean spectrum (and standard deviation of ln Sa) given Sa(Tstar) = im.

    Computed scenario by scenario (Lin et al., 2013): scenario i reaches im with
    epsilon_i = (ln im - mean_i) / sigma_i; given that, the expected ln Sa(T) is
    mean_i(T) + rho(T, Tstar) epsilon_i sigma_i(T). Scenarios are weighted by how often they
    produce Sa(Tstar) = im, i.e. lambda(rup_i) times the probability density at im."""
    sc = scenarios(p) if sc is None else sc
    gm = lambda T: cy14(T, sc["mag"], sc["rrup"], sc["rjb"], sc["ztor"], p["vs30"])
    mean, sigma = gm(Tstar)
    eps = (np.log(im) - mean) / sigma
    w = sc["rate"] * np.exp(-eps ** 2 / 2) / sigma * (np.abs(eps) <= p["trunc"])
    w = w * np.isin(sc["source"], sources)
    w = w / w.sum()
    mu, sd = [], []
    for T in periods:
        rho = correlation(T, Tstar)
        m, s = gm(T)
        mi = m + rho * eps * s                      # conditional mean, scenario by scenario
        vi = s ** 2 * (1 - rho ** 2)                # conditional variance
        mu.append(np.sum(w * mi))
        sd.append(np.sqrt(np.sum(w * (vi + mi ** 2)) - mu[-1] ** 2))
    return np.exp(mu), np.array(sd)
