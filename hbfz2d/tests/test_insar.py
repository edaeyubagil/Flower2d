"""MintPy InSAR okuma uzlaşımları (sentetik HDF5 ile)."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from profil2d.forward import los_unit_vector
from profil2d.insar import (los_unit_vector_mintpy, read_mintpy, block_median, local_anomaly_mask,
                            empirical_semivariogram, fit_exponential)


@pytest.mark.parametrize("inc, heading", [(41.0, -13.44), (37.3, -166.57), (35.0, 190.0)])
def test_mintpy_azimuth_convention(inc, heading):
    """MintPy azimuthAngle: yer->uydu yönü, kuzeyden SAAT YÖNÜNÜN TERSİNE. Eşdeğeri: heading = 90 - azimuthAngle."""
    alpha = 90.0 - heading
    assert np.allclose(los_unit_vector_mintpy(inc, alpha), los_unit_vector(inc, heading), atol=1e-12)


def test_matches_mintpy_enu2los_verbatim():
    """MintPy 1.6.4 utils0.enu2los (satır satır kopya): + = uyduya doğru."""
    def enu2los(v_e, v_n, v_u, inc_angle, az_angle):
        return (v_e * np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
                + v_n * np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
                + v_u * np.cos(np.deg2rad(inc_angle)))
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(0, 5, 3); inc = rng.uniform(25, 48); az = rng.uniform(-180, 180)
        assert los_unit_vector_mintpy(inc, az) @ v == pytest.approx(enu2los(*v, inc, az), abs=1e-12)


def test_mintpy_unit_vector_signs_for_real_values():
    asc = los_unit_vector_mintpy(41.0, 99.8)      # dosyadaki tipik yükselen değer
    dsc = los_unit_vector_mintpy(37.3, -100.3)    # dosyadaki tipik alçalan değer
    assert asc[0] < -0.6 and dsc[0] > 0.55        # yükselen: uydu batıda; alçalan: doğuda
    assert asc[1] < 0 and dsc[1] < 0              # ikisinde de küçük güney bileşeni
    assert asc[2] > 0 and dsc[2] > 0


@pytest.fixture
def mintpy_pair(tmp_path):
    ny, nx = 4, 5
    attrs = dict(X_FIRST="500000.0", Y_FIRST="4400000.0", X_STEP="80.0", Y_STEP="-80.0", EPSG="32635",
                 UNIT="m/year", ORBIT_DIRECTION="DESCENDING", HEADING="-166.57", WIDTH=str(nx), LENGTH=str(ny))
    vel = np.full((ny, nx), 0.002, np.float32); vel[0, 0] = 0.0          # 0 = veri yok
    std = np.full((ny, nx), 0.0005, np.float32)
    inc = np.full((ny, nx), 37.0, np.float32); az = np.full((ny, nx), -100.0, np.float32)
    water = np.ones((ny, nx), bool); water[3, 4] = False                # False = su
    pv, pg = tmp_path / "v.h5", tmp_path / "g.h5"
    with h5py.File(pv, "w") as f:
        f["velocity"] = vel; f["velocityStd"] = std; f.attrs.update(attrs)
    with h5py.File(pg, "w") as f:
        f["incidenceAngle"] = inc; f["azimuthAngle"] = az; f["waterMask"] = water; f.attrs.update(attrs)
    return str(pv), str(pg)


def test_read_mintpy_units_mask_and_coordinates(mintpy_pair):
    d = read_mintpy(*mintpy_pair)
    assert d["los"].shape == (4, 5)
    assert d["los"][1, 1] == pytest.approx(2.0)            # m/yıl -> mm/yıl
    assert d["std"][1, 1] == pytest.approx(0.5)
    assert np.isnan(d["los"][0, 0])                         # sıfır -> NaN
    assert np.isnan(d["los"][3, 4])                         # su -> NaN
    # piksel merkezi = ilk köşe + (i + 0.5) * adım, UTM35 -> boylam/enlem
    import pyproj
    lon, lat = pyproj.Transformer.from_crs(32635, 4326, always_xy=True).transform(500000 + 2.5 * 80, 4400000 - 1.5 * 80)
    assert d["lon"][1, 2] == pytest.approx(lon, abs=1e-9) and d["lat"][1, 2] == pytest.approx(lat, abs=1e-9)
    assert d["orbit"] == "DESCENDING"


def test_block_median_ignores_nan_and_requires_coverage():
    a = np.arange(16, dtype=float).reshape(4, 4); a[0, 0] = np.nan
    m, n = block_median(a, 2, min_frac=0.5)
    assert m.shape == (2, 2) and m[0, 0] == pytest.approx(np.median([1, 4, 5])) and n[0, 0] == 3
    a[0:2, 0:2] = np.nan; a[0, 1] = 7.0
    m, n = block_median(a, 2, min_frac=0.5)
    assert np.isnan(m[0, 0]) and n[0, 0] == 1               # %25 kapsama < %50 -> NaN


def test_local_anomaly_mask_finds_bowl_not_gradient():
    yy, xx = np.mgrid[0:60, 0:60].astype(float)
    g = 0.05 * xx - 0.03 * yy                                   # yumuşak bölgesel alan (km başına mm/yıl)
    g[30:34, 30:34] -= 20.0                                     # 4x4 km çökme çanağı
    g[5, 50] = np.nan
    m = local_anomaly_mask(g, window=15, thresh=3.0, dilate=1)
    assert m[30:34, 30:34].all() and m[29, 29] and m[34, 34]    # çanak + 1 blok tampon
    assert m.sum() <= 6 * 6 + 1                                 # gradyan maskelenmiyor
    assert not m[10, 10]


def test_semivariogram_recovers_exponential_parameters():
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 120, 900), rng.uniform(0, 120, 900)
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    sig, lam, nug = 2.0, 15.0, 0.3
    C = sig ** 2 * np.exp(-d / lam) + nug ** 2 * np.eye(len(x))
    z = np.linalg.cholesky(C) @ rng.normal(size=len(x))
    h, gam, npair = empirical_semivariogram(x, y, z, bins=np.arange(0, 60, 3.0), max_pairs=None)
    p = fit_exponential(h, gam, npair)
    assert p["sigma"] == pytest.approx(sig, rel=0.35) and p["lambda"] == pytest.approx(lam, rel=0.5)
