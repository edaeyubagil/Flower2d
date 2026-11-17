"""GLOBK .org okuyucu ve GPS kalite kontrol yardımcıları (sentetik veriyle)."""
import numpy as np
import pytest

from profil2d.gps import read_globk_org, neighbor_outliers, collocated_pairs

ORG = """Globk Analysis
 SUMMARY VELOCITY ESTIMATES FROM GLOBK Ver 5.35
   Long.       Lat.         E & N Rate      E & N Adj.      E & N +-   RHO        H Rate   H adj.    +-  SITE
   (deg)      (deg)           (mm/yr)       (mm/yr)       (mm/yr)                 (mm/yr)
  356.00000   40.00000    -1.00   -0.50   -0.60   -0.50    0.02    0.01  0.200     -0.40    1.80    0.05 AAAA_GPS*
   27.50000   39.50000   -20.00   -8.00    0.00    0.00    0.30    0.40  0.010      3.00    0.00    1.50 BBBB_GPS

 VEL STATISTICS: something
"""


def test_read_globk_org(tmp_path):
    f = tmp_path / "x.org"; f.write_text(ORG)
    g = read_globk_org(str(f))
    assert list(g["site"]) == ["AAAA", "BBBB"]
    assert g["lon"][0] == pytest.approx(-4.0)             # 356 -> -4
    assert g["ve"][1] == pytest.approx(-20.0) and g["vn"][1] == pytest.approx(-8.0)
    assert g["se"][1] == pytest.approx(0.30) and g["rho"][1] == pytest.approx(0.010)
    assert g["vu"][1] == pytest.approx(3.0) and g["su"][1] == pytest.approx(1.50)


def test_neighbor_outlier_detects_injected_station():
    rng = np.random.default_rng(1)
    e = rng.uniform(-60, 60, 120); n = rng.uniform(-60, 60, 120)
    ve = 0.02 * n + rng.normal(0, 0.3, 120); vn = -0.01 * e + rng.normal(0, 0.3, 120)
    ve[17] += 5.0
    flag, res_e, res_n, scale = neighbor_outliers(e, n, ve, vn, radius=25.0, nsig=3.0)
    assert flag[17] and flag.sum() <= 3


def test_collocated_pairs():
    e = np.array([0.0, 0.8, 50.0]); n = np.array([0.0, 0.5, 0.0])
    assert collocated_pairs(e, n, maxdist=2.0) == [(0, 1)]
