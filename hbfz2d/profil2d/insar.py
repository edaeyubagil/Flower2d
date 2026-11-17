"""MintPy InSAR ürünlerini okuma.

Uzlaşımlar (tests/test_insar.py ile kilitli)
--------------------------------------------
* Hız birimi m/yıl -> mm/yıl'a çevrilir; 0 değerleri ve su pikselleri (waterMask == False) NaN olur.
* MintPy azimuthAngle: yer -> uydu vektörünün yatay yönü, kuzeyden SAAT YÖNÜNÜN TERSİNE (derece).
  Bizim forward.los_unit_vector'daki uçuş yönü eşdeğeri: heading = 90 - azimuthAngle.
* LOS işareti dosyada olduğu gibi kalır (MintPy: pozitif = uyduya doğru). Bu varsayım
  scripts/insar_kontrol.py ile GPS'e karşı veriyle sınanır.
* Piksel merkezi = X_FIRST + (j + 0.5) * X_STEP (MintPy/GDAL köşe uzlaşımı).
"""
import numpy as np


def los_unit_vector_mintpy(incidence, azimuth_angle):
    """Yer -> uydu birim vektörü (doğu, kuzey, yukarı), MintPy açılarından."""
    inc = np.deg2rad(np.asarray(incidence, float)); a = np.deg2rad(np.asarray(azimuth_angle, float))
    return np.array([-np.sin(inc) * np.sin(a), np.sin(inc) * np.cos(a), np.cos(inc) * np.ones_like(a)])


def _attr(attrs, k, default=None):
    v = attrs.get(k, default)
    return v.decode() if isinstance(v, bytes) else v


def read_mintpy(velocity_path, geometry_path, dataset="velocity"):
    """Hız + geometri dosyalarını okur. Dönüş: dict(lon, lat, los, std, incidence, azimuth, orbit, heading, ref, epsg)."""
    import h5py, pyproj
    with h5py.File(velocity_path, "r") as f:
        a = dict(f.attrs)
        los = f[dataset][...].astype(float)
        std = f[dataset + "Std"][...].astype(float) if (dataset + "Std") in f else np.full(los.shape, np.nan)
    with h5py.File(geometry_path, "r") as g:
        inc = g["incidenceAngle"][...].astype(float); az = g["azimuthAngle"][...].astype(float)
        water = g["waterMask"][...].astype(bool) if "waterMask" in g else np.ones(los.shape, bool)
    unit = _attr(a, "UNIT", "m/year").lower()
    fac = 1000.0 if unit.startswith("m/") or unit == "m" else 1.0
    bad = (los == 0) | ~np.isfinite(los) | ~water | ~np.isfinite(inc)
    los = np.where(bad, np.nan, los * fac); std = np.where(bad, np.nan, std * fac)
    ny, nx = los.shape
    x0, y0 = float(_attr(a, "X_FIRST")), float(_attr(a, "Y_FIRST"))
    dx, dy = float(_attr(a, "X_STEP")), float(_attr(a, "Y_STEP"))
    xx, yy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * dx, y0 + (np.arange(ny) + 0.5) * dy)
    epsg = _attr(a, "EPSG")
    if epsg is not None and int(epsg) != 4326:
        lon, lat = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True).transform(xx, yy)
    else:
        lon, lat = xx, yy
    ref = None
    if _attr(a, "REF_Y") is not None:
        ry, rx = int(_attr(a, "REF_Y")), int(_attr(a, "REF_X"))
        ref = (float(lon[ry, rx]), float(lat[ry, rx]))
    return dict(lon=np.asarray(lon), lat=np.asarray(lat), los=los, std=std, incidence=np.where(bad, np.nan, inc),
                azimuth=np.where(bad, np.nan, az), orbit=_attr(a, "ORBIT_DIRECTION"),
                heading=float(_attr(a, "HEADING", "nan")), ref=ref, epsg=epsg)


def block_median(a, k, min_frac=0.5):
    """k x k bloklarda NaN'ları yok sayan medyan; geçerli piksel oranı < min_frac ise NaN. Dönüş: (medyan, sayı)."""
    a = np.asarray(a, float)
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k).transpose(0, 2, 1, 3).reshape(ny // k, nx // k, k * k)
    n = np.sum(np.isfinite(b), axis=2)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(b, axis=2)
    m[n < min_frac * k * k] = np.nan
    return m, n


def local_anomaly_mask(grid, window=15, thresh=3.0, dilate=1):
    """Yerel (tektonik olmayan) anomaliler: |grid − kayan nanmedyan(window)| > thresh, ardından
    `dilate` blokluk tampon. NaN hücreler anomali sayılmaz. Dönüş: bool maske (True = at)."""
    from scipy.ndimage import generic_filter, binary_dilation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = generic_filter(np.asarray(grid, float), np.nanmedian, size=window, mode="nearest")
    with np.errstate(invalid="ignore"):
        m = np.abs(grid - ref) > thresh
    if dilate:
        m = binary_dilation(m, structure=np.ones((3, 3), bool), iterations=dilate)
    return m


def empirical_semivariogram(x, y, z, bins, max_pairs=300000, seed=0):
    """γ(h) = ½·ortalama((z_i − z_j)²) mesafe bölmelerinde. max_pairs verilirse rastgele çift örneklemesi.
    Dönüş: (ortalama mesafe, γ, çift sayısı) — boş bölmeler atılır."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    n = len(z)
    if max_pairs is None or n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs); k = i != j; i, j = i[k], j[k]
    d = np.hypot(x[i] - x[j], y[i] - y[j]); g = 0.5 * (z[i] - z[j]) ** 2
    idx = np.digitize(d, bins) - 1
    h, gam, npair = [], [], []
    for b in range(len(bins) - 1):
        s = idx == b
        if s.sum() > 0:
            h.append(d[s].mean()); gam.append(g[s].mean()); npair.append(int(s.sum()))
    return np.array(h), np.array(gam), np.array(npair)


def fit_exponential(h, gam, npair):
    """γ(h) = nugget² + σ²·(1 − exp(−h/λ)) ağırlıklı (çift sayısı) uyumu. Dönüş: dict(sigma, lambda, nugget).
    Karşılık gelen kovaryans: C(h) = σ²·exp(−h/λ) (+ nugget² köşegende)."""
    from scipy.optimize import curve_fit
    f = lambda hh, s, l, nu: nu ** 2 + s ** 2 * (1 - np.exp(-hh / l))
    p0 = [np.sqrt(max(gam.max(), 1e-6)), max(h.max() / 4, 1e-3), np.sqrt(max(gam.min(), 1e-6))]
    p, _ = curve_fit(f, h, gam, p0=p0, sigma=1 / np.sqrt(npair), bounds=([0, 1e-3, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
    return dict(sigma=float(p[0]), **{"lambda": float(p[1])}, nugget=float(p[2]))
