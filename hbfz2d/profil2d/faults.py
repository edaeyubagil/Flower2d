"""Fay izleri: GeoJSON LineString okuma, baskın doğrultu ve profil üzerindeki konumlar."""
import json
import numpy as np

from .geometry import Profile


def read_traces(path, name_key="FAY_ADI"):
    """{fay_adı: [ (k,2) dizi (lon, lat), ... ]} — z koordinatı atılır."""
    out = {}
    for f in json.load(open(path))["features"]:
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        name = (f.get("properties") or {}).get(name_key, "fay")
        for c in parts:
            out.setdefault(name, []).append(np.asarray(c, float)[:, :2])
    return out


def _all_vertices(traces):
    return np.vstack([c for parts in traces.values() for c in parts])


def principal_strike(traces):
    """Segment azimutlarının uzunlukla ağırlıklı eksenel ortalaması (açı ikiye katlanarak).

    Azimut [0, 180) aralığında döner; sayısallaştırma yönünden ve paralel fayların birbirine göre
    kaymasından bağımsızdır (köşe PCA'sı bunun yerine zonun uzanım yönünü verirdi)."""
    v = _all_vertices(traces)
    p = Profile(v[:, 0].mean(), v[:, 1].mean(), 0.0)
    c2 = s2 = 0.0
    for parts in traces.values():
        for c in parts:
            e, n = p.to_local(c[:, 0], c[:, 1])
            de, dn = np.diff(e), np.diff(n)
            L = np.hypot(de, dn); a = np.arctan2(de, dn)
            c2 += np.sum(L * np.cos(2 * a)); s2 += np.sum(L * np.sin(2 * a))
    return float(np.degrees(0.5 * np.arctan2(s2, c2)) % 180.0)


def centroid(traces):
    v = _all_vertices(traces)
    return float(v[:, 0].mean()), float(v[:, 1].mean())


def fault_offsets(traces, profile, xmax=None):
    """Her fayın profil üzerindeki ortalama y konumu (km); xmax verilirse |x| <= xmax köşeler."""
    out = {}
    for name, parts in traces.items():
        v = np.vstack(parts)
        x, y = profile.to_profile(v[:, 0], v[:, 1])
        k = np.ones(len(x), bool) if xmax is None else np.abs(x) <= xmax
        out[name] = float(np.mean(y[k])) if k.any() else float("nan")
    return out
