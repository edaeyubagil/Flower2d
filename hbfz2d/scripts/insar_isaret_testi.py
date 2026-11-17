#!/usr/bin/env python3
"""Rampadan bağımsız LOS işaret testi: yerel anomaliler.

InSAR anomalisi = medyan(1 km) − medyan(3–15 km halka)
GPS anomalisi   = p · (v_i − medyan(v_komşu, 30 km))  [3B; p piksel geometrisinden]
Pozitif = uyduya doğru ise eğim ≈ +1 (düşey baskın yerel sinyallerde de).
Dışlanan (yerel etkili) istasyonlar BU testte bilerek kullanılır; yalnızca σ'sı büyük olanlar atılır.

Kullanım: python scripts/insar_isaret_testi.py --org VEL.org --track AD HIZ GEOMETRI [--track ...] [--fig F]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profil2d.gps import read_globk_org
from profil2d.insar import read_mintpy, los_unit_vector_mintpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--track", nargs=3, action="append", metavar=("AD", "HIZ", "GEOMETRI"), required=True)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    import pyproj
    g = read_globk_org(a.org)
    ok = (g["se"] < 2) & (g["sn"] < 2) & (g["su"] < 3)
    tr = pyproj.Transformer.from_crs(4326, 32635, always_xy=True)
    gx, gy = tr.transform(g["lon"], g["lat"]); gx, gy = gx / 1e3, gy / 1e3
    out = []
    for name, vpath, gpath in a.track:
        d = read_mintpy(vpath, gpath)
        px, py = tr.transform(d["lon"], d["lat"]); px, py = px / 1e3, py / 1e3
        v = np.isfinite(d["los"]); pv, qv = px[v], py[v]; tree = cKDTree(np.c_[pv, qv])
        los, inc, az = d["los"][v], d["incidence"][v], d["azimuth"][v]
        gtree = cKDTree(np.c_[gx[ok], gy[ok]]); idx_ok = np.flatnonzero(ok)
        rows = []
        for i in idx_ok:
            near = tree.query_ball_point([gx[i], gy[i]], 1.0)
            ring = []
            if len(near) >= 30:
                big = np.asarray(tree.query_ball_point([gx[i], gy[i]], 15.0))
                ring = big[np.hypot(pv[big] - gx[i], qv[big] - gy[i]) >= 3.0]
            nb = [idx_ok[k] for k in gtree.query_ball_point([gx[i], gy[i]], 30.0) if idx_ok[k] != i]
            if len(near) >= 30 and len(ring) >= 300 and len(nb) >= 3:
                p = los_unit_vector_mintpy(np.median(inc[near]), np.median(az[near]))
                dv = np.array([g["ve"][i] - np.median(g["ve"][nb]), g["vn"][i] - np.median(g["vn"][nb]), g["vu"][i] - np.median(g["vu"][nb])])
                rows.append((g["site"][i], p @ dv, dv[2] * p[2], np.median(los[near]) - np.median(los[ring])))
        s = np.array([r[0] for r in rows]); pred = np.array([r[1] for r in rows]); predU = np.array([r[2] for r in rows]); obs = np.array([r[3] for r in rows])
        b = np.polyfit(pred, obs, 1)[0]; r = np.corrcoef(pred, obs)[0, 1]
        big = np.abs(pred) > 3
        bb = np.polyfit(pred[big], obs[big], 1)[0] if big.sum() >= 3 else np.nan
        sgn = np.mean(np.sign(pred[big]) == np.sign(obs[big])) if big.any() else np.nan
        print(f"=== {name}: {len(rows)} istasyon; eğim {b:+.2f}, r = {r:+.2f};  |öngörü|>3 mm/yıl olan {big.sum()} istasyonda eğim {bb:+.2f}, işaret uyumu {sgn:.0%}")
        for k in np.argsort(-np.abs(pred))[:8]:
            print(f"     {s[k]:5s} GPS→LOS anomali {pred[k]:+6.1f} (düşey katkı {predU[k]:+6.1f})   InSAR anomali {obs[k]:+6.1f} mm/yıl")
        out.append((name, pred, obs, b, r))
    if a.fig and out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(out), figsize=(5.2 * len(out), 4.8)); ax = np.atleast_1d(ax)
        for k, (name, pred, obs, b, r) in enumerate(out):
            lim = np.array([min(pred.min(), obs.min()) - 1, max(pred.max(), obs.max()) + 1])
            ax[k].scatter(pred, obs, s=16, color="#1f6feb"); ax[k].plot(lim, lim, "k--", lw=.8, label="1:1")
            ax[k].plot(lim, b * lim, color="#1a7f37", label=f"eğim {b:.2f}, r={r:.2f}")
            ax[k].set_xlabel("GPS yerel anomali → LOS (mm/yıl)"); ax[k].set_ylabel("InSAR yerel anomali (mm/yıl)")
            ax[k].set_title(f"{name}: + = uyduya doğru varsayımı"); ax[k].legend(fontsize=8); ax[k].grid(alpha=.3); ax[k].set_aspect("equal")
        fig.tight_layout(); fig.savefig(a.fig, dpi=120); print("figür:", a.fig)


if __name__ == "__main__":
    main()
