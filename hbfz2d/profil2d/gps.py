"""GPS girdi/çıktı ve kalite kontrol yardımcıları."""
import numpy as np


def read_globk_org(path):
    """GLOBK/GLORG .org dosyasındaki 'SUMMARY VELOCITY ESTIMATES' tablosunu okur.

    Dönüş: dict(site, lon, lat, ve, vn, se, sn, rho, vu, su) — hızlar mm/yıl, boylam [-180, 180).
    Çerçeve dosyada nasıl tanımlıysa öyledir (ör. EUR14 = Avrasya sabit)."""
    lines = open(path).read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if "SUMMARY VELOCITY ESTIMATES" in l)
    rows = []
    for l in lines[i0 + 1:]:
        t = l.split()
        if len(t) == 13:
            try:
                float(t[0]); rows.append(t)
            except ValueError:
                continue
        elif rows:
            break
    col = lambda k: np.array([float(r[k]) for r in rows])
    lon = col(0)
    return dict(site=np.array([r[12].rstrip("*").split("_")[0] for r in rows]),
                lon=np.where(lon >= 180, lon - 360, lon), lat=col(1), ve=col(2), vn=col(3),
                se=col(6), sn=col(7), rho=col(8), vu=col(9), su=col(11))


def collocated_pairs(e, n, maxdist=2.0):
    """Birbirine maxdist km'den yakın istasyon çiftlerinin indeksleri."""
    out = []
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            if np.hypot(e[i] - e[j], n[i] - n[j]) < maxdist:
                out.append((i, j))
    return out


def neighbor_outliers(e, n, ve, vn, radius=25.0, nsig=3.0, min_neighbors=3, exclude=None):
    """Her istasyonu radius km içindeki komşularının medyan hızıyla karşılaştırır.

    Ölçek = komşu rezidüellerinin robust std'si (1.4826*MAD), bileşen başına.
    Eş konumlu (< 2 km) ikizler birbirinin komşusu sayılmaz. Dönüş: (bayrak, res_e, res_n, (ölçek_e, ölçek_n))."""
    e, n, ve, vn = map(np.asarray, (e, n, ve, vn))
    use = np.ones(len(e), bool) if exclude is None else ~np.asarray(exclude, bool)
    twins = {i: set() for i in range(len(e))}
    for i, j in collocated_pairs(e, n):
        twins[i].add(j); twins[j].add(i)
    re_, rn_ = np.full(len(e), np.nan), np.full(len(e), np.nan)
    for i in np.flatnonzero(use):
        d = np.hypot(e - e[i], n - n[i])
        k = [j for j in np.flatnonzero(use & (d < radius) & (d > 0)) if j not in twins[i]]
        if len(k) >= min_neighbors:
            re_[i] = ve[i] - np.median(ve[k]); rn_[i] = vn[i] - np.median(vn[k])
    ok = ~np.isnan(re_)
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    sc = (mad(re_[ok]), mad(rn_[ok]))
    flag = ok & ((np.abs(re_) > nsig * sc[0]) | (np.abs(rn_) > nsig * sc[1]))
    return flag, re_, rn_, sc
