"""The course's toy PSHA model, built with OpenQuake hazardlib.

The same model as psha_toy.py (same parameters, PARAMS), expressed as OpenQuake sources:
  * the fault as a SimpleFaultSource with a Youngs & Coppersmith (1985) MFD balanced to the
    fault's slip rate, ruptures sized by Wells & Coppersmith (1994);
  * the area source as an AreaSource with a truncated Gutenberg-Richter MFD and point
    ruptures at a fixed depth;
  * the Chiou & Youngs (2014) ground-motion model.

OpenQuake works in longitude/latitude: the site is placed at (0, 0) and the model's flat
coordinates (km) are converted near the equator, where 1 degree = 111.195 km.
"""
import numpy as np
from openquake.hazardlib.geo import Point, Line, Polygon, NodalPlane
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source import SimpleFaultSource, AreaSource
from openquake.hazardlib.mfd import YoungsCoppersmith1985MFD, TruncatedGRMFD
from openquake.hazardlib.scalerel import WC1994, PointMSR
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
from openquake.hazardlib.calc.hazard_curve import calc_hazard_curves
from openquake.hazardlib.calc.filters import SourceFilter, IntegrationDistance
from openquake.hazardlib.sourceconverter import SourceGroup

from psha_toy import PARAMS

TRT = "Active Shallow Crust"
KM_PER_DEG = 111.19492664455873        # OpenQuake's Earth radius x pi / 180


def lonlat(x, y):
    """Flat coordinates (km, site at the origin) to longitude/latitude near the equator."""
    return x / KM_PER_DEG, y / KM_PER_DEG


def fault_source(p=PARAMS):
    y = p["fault_dist"]
    trace = Line([Point(*lonlat(-p["fault_left"], y)), Point(*lonlat(p["fault_right"], y))])
    area = (p["fault_left"] + p["fault_right"]) * (p["fault_bottom"] - p["fault_top"]) * 1e6   # m^2
    moment_rate = p["shear_modulus"] * area * p["fault_slip_rate"] / 1000.0                  # N m / yr
    mfd = YoungsCoppersmith1985MFD.from_total_moment_rate(
        p["fault_mmin"], p["fault_b"], p["fault_mchar"], moment_rate, p["dm"])
    return SimpleFaultSource(
        "fault", "Fault", TRT, mfd, rupture_mesh_spacing=p["mesh"],
        magnitude_scaling_relationship=WC1994(), rupture_aspect_ratio=p["aspect"],
        temporal_occurrence_model=PoissonTOM(1.0),
        upper_seismogenic_depth=p["fault_top"], lower_seismogenic_depth=p["fault_bottom"],
        fault_trace=trace, dip=90.0, rake=0.0)


def area_source(p=PARAMS):
    t = np.radians(np.arange(360.0))
    R = p["area_radius"]
    polygon = Polygon([Point(*lonlat(R * np.cos(a), R * np.sin(a))) for a in t])
    b, m0, m1 = p["area_b"], p["area_mmin"], p["area_mmax"]
    a = np.log10(p["area_rate"] / (10 ** (-b * m0) - 10 ** (-b * m1)))    # N(M >= m0) = area_rate
    return AreaSource(
        "area", "Area", TRT, TruncatedGRMFD(m0, m1, p["dm"], a, b),
        rupture_mesh_spacing=p["mesh"], magnitude_scaling_relationship=PointMSR(),
        rupture_aspect_ratio=1.0, temporal_occurrence_model=PoissonTOM(1.0),
        upper_seismogenic_depth=0.0, lower_seismogenic_depth=p["fault_bottom"],
        nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90.0, 0.0))]),
        hypocenter_distribution=PMF([(1.0, p["area_depth"])]),
        polygon=polygon, area_discretization=p["area_grid"])


def site(p=PARAMS):
    # z1pt0 = -999: basin depth set to its average for the Vs30 (CY14's default)
    return SiteCollection([Site(Point(0.0, 0.0), vs30=p["vs30"], vs30measured=True, z1pt0=-999)])


def imt_name(T):
    return "PGA" if T == 0 else f"SA({T})"


def hazard_curves(periods, imls, p=PARAMS, sources=("fault", "area")):
    """Annual rates of exceedance lambda(IM > im) for each period, from OpenQuake.

    OpenQuake returns probabilities of exceedance in one year, PoE = 1 - exp(-lambda);
    these are converted back to rates."""
    srcs = [s for s in (fault_source(p), area_source(p)) if s.source_id in sources]
    imtls = {imt_name(T): np.asarray(imls, float) for T in periods}
    poes = calc_hazard_curves([SourceGroup(TRT, srcs)],
                              SourceFilter(site(p), IntegrationDistance.new("300")),
                              imtls, {TRT: ChiouYoungs2014()}, truncation_level=p["trunc"])
    return {T: -np.log1p(-poes[imt_name(T)][0]) for T in periods}
