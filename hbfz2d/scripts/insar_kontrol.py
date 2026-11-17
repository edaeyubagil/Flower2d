#!/usr/bin/env python3
"""InSAR işaret/geometri kontrolü: GPS hızlarını LOS'a izdüşür, aynı yerdeki InSAR medyanıyla karşılaştır.

insar = c0 + c1*e + c2*n + b * (GPS · p)   (ofset + düzlemsel rampa referans/çerçeve farkını soğurur)
b ≈ +1  ->  işaret (pozitif = uyduya doğru) ve azimut tanımı doğru.

Kullanım:
  python scripts/insar_kontrol.py --org VEL.org --haric local/gps_haric.txt \
      --track asc local/insar_raw/velocity_..._asc.h5 local/insar_raw/geometryGeo_asc.h5 \
      --track dsc local/insar_raw/velocity_..._dsc.h5 local/insar_raw/geometryGeo_dsc.h5 \
      [--radius 1.0] [--minpix 30] [--fig local/insar_kontrol.png]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profil2d.gps import read_globk_org
from profil2d.insar import read_mintpy, los_unit_vector_mintpy
from profil2d.forward import los_unit_vector


def fit(pred, obs, e, n):
    A = np.c_[np.ones_like(pred), e, n, pred]
    keep = np.ones(len(obs), bool)
    for _ in range(2):                                   # bir tur 3σ ayıklama
        c, *_ = np.linalg.lstsq(A[keep], obs[keep], rcond=None)
        r = obs - A @ c; s = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < 3 * max(s, 1e-6)
    c, *_ = np.linalg.lstsq(A[keep], obs[keep], rcond=None)
    r = obs[keep] - A[keep] @ c; dof = keep.sum() - 4
    cov = np.linalg.inv(A[keep].T @ A[keep]) * (r @ r) / max(dof, 1)
    ramp = A[keep][:, :3] @ c[:3]
    rr = np.corrcoef(pred[keep], obs[keep] - ramp)[0, 1]
    return c[3], np.sqrt(cov[3, 3]), rr, keep.sum(), np.sqrt(np.mean(r ** 2)), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True); ap.add_argument("--haric", required=True)
    ap.add_argument("--track", nargs=3, action="append", metavar=("AD", "HIZ", "GEOMETRI"), required=True)
    ap.add_argument("--radius", type=float, default=1.0); ap.add_argument("--minpix", type=int, default=30)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    import pyproj

    g = read_globk_org(a.org)
    excl = {l.split()[0] for l in open(a.haric) if l.strip() and not l.startswith("#")}
    ok = ~np.isin(g["site"], list(excl)) & (g["se"] < 2) & (g["sn"] < 2)
    figs = []
    for name, vpath, gpath in a.track:
        d = read_mintpy(vpath, gpath)
        tr = pyproj.Transformer.from_crs(4326, 32635, always_xy=True)
        px, py = tr.transform(d["lon"], d["lat"]); px, py = px / 1e3, py / 1e3
        valid = np.isfinite(d["los"])
        tree = cKDTree(np.c_[px[valid], py[valid]])
        los, inc, az = d["los"][valid], d["incidence"][valid], d["azimuth"][valid]
        gx, gy = tr.transform(g["lon"], g["lat"]); gx, gy = gx / 1e3, gy / 1e3
        rows = []
        for i in np.flatnonzero(ok):
            idx = tree.query_ball_point([gx[i], gy[i]], a.radius)
            if len(idx) >= a.minpix:
                rows.append((i, np.median(los[idx]), np.median(inc[idx]), np.median(az[idx]), len(idx)))
        if not rows:
            print(f"{name}: GPS çevresinde yeterli piksel yok"); continue
        I = np.array([r[0] for r in rows]); obs = np.array([r[1] for r in rows])
        inc_s, az_s = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
        ve, vn, vu = g["ve"][I], g["vn"][I], g["vu"][I]
        print(f"\n=== {name} ({d['orbit']}, HEADING {d['heading']:.2f}°): {len(I)} temiz GPS istasyonunda ≥{a.minpix} piksel ({a.radius} km)")
        print(f"    referans noktası: {d['ref'][0]:.3f}D {d['ref'][1]:.3f}K")
        p_ok = los_unit_vector_mintpy(inc_s, az_s)
        p_cw = np.array([np.sin(np.deg2rad(inc_s)) * np.sin(np.deg2rad(az_s)), np.sin(np.deg2rad(inc_s)) * np.cos(np.deg2rad(az_s)), np.cos(np.deg2rad(inc_s))])
        p_hd = los_unit_vector(inc_s, np.full_like(inc_s, d["heading"]))
        cands = [("MintPy azimutu (saat tersi), yalnız yatay GPS", p_ok, False),
                 ("MintPy azimutu, yatay + düşey GPS", p_ok, True),
                 ("YANLIŞ yorum: azimut saat yönünde", p_cw, False),
                 ("HEADING özniteliği (sabit), yalnız yatay", p_hd, False)]
        e_, n_ = gx[I] - gx[I].mean(), gy[I] - gy[I].mean()
        for lab, p, use_u in cands:
            pred = ve * p[0] + vn * p[1] + (vu * p[2] if use_u else 0)
            b, sb, rr, nk, rms, keep = fit(pred, obs, e_, n_)
            print(f"    {lab:46s} eğim b = {b:+.2f} ± {sb:.2f}   r = {rr:+.2f}   (n={nk}, rezidüel RMS {rms:.2f} mm/yıl)")
            if lab.startswith("MintPy azimutu (saat"):
                figs.append((name, pred, obs, e_, n_, keep, b))
        print(f"    birim vektör medyanı (D,K,Y) = ({np.median(p_ok[0]):+.3f}, {np.median(p_ok[1]):+.3f}, {np.median(p_ok[2]):+.3f})")
    if a.fig and figs:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(figs), figsize=(5.2 * len(figs), 4.6))
        ax = np.atleast_1d(ax)
        for k, (name, pred, obs, e_, n_, keep, b) in enumerate(figs):
            A = np.c_[np.ones_like(pred), e_, n_, pred]; c, *_ = np.linalg.lstsq(A[keep], obs[keep], rcond=None)
            o = obs - A[:, :3] @ c[:3]
            ax[k].scatter(pred[keep], o[keep], s=14, color="#1f6feb", label="kullanılan")
            ax[k].scatter(pred[~keep], o[~keep], s=18, marker="x", color="#d1242f", label="3σ dışı")
            lim = np.array([min(pred.min(), o.min()), max(pred.max(), o.max())])
            ax[k].plot(lim, lim, "k--", lw=0.8, label="1:1"); ax[k].plot(lim, b * lim, "-", color="#1a7f37", label=f"eğim {b:.2f}")
            ax[k].set_xlabel("GPS → LOS (yatay) mm/yıl"); ax[k].set_ylabel("InSAR − (ofset+rampa) mm/yıl")
            ax[k].set_title(name); ax[k].legend(fontsize=8); ax[k].grid(alpha=.3); ax[k].set_aspect("equal")
        fig.tight_layout(); fig.savefig(a.fig, dpi=120); print("\nfigür:", a.fig)


if __name__ == "__main__":
    main()
