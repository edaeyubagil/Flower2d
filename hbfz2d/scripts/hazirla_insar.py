#!/usr/bin/env python3
"""MintPy hızlarını 2B profil modeline hazırlar (Aşama 1, InSAR).

Kullanım:
  python scripts/hazirla_insar.py --profil local/gps_profil.csv --faults FAYLAR.geojson \
      --track asc HIZ_asc.h5 GEO_asc.h5 --track dsc HIZ_dsc.h5 GEO_dsc.h5 --outdir local \
      [--block 12] [--stdmax 2] [--window 15] [--thresh 3] [--xmax 50] [--ymax 100]

Çıktı (iz başına): <outdir>/insar_<ad>_profil.csv  (lon,lat,x_km,y_km,los,npix,pE,pN,pU)
                   <outdir>/insar_<ad>_*.png, yarıvariogram parametreleri ekrana.
LOS: mm/yıl, pozitif = uyduya doğru (MintPy). Hızlardan MintPy'de doğrusal rampa çıkarılmıştır
(mintpy.deramp=linear); modelde iz başına ofset + rampa kestirilmelidir.
"""
import argparse, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profil2d.geometry import Profile
from profil2d.faults import read_traces, fault_offsets
from profil2d.insar import (read_mintpy, block_median, local_anomaly_mask, los_unit_vector_mintpy,
                            empirical_semivariogram, fit_exponential)


def profile_from_header(path):
    h = open(path).readline()
    g = lambda k: float(re.search(rf"{k}=([-\d.]+)", h).group(1))
    return Profile(g("merkez_lon"), g("merkez_lat"), g("strike"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profil", required=True); ap.add_argument("--faults", required=True)
    ap.add_argument("--track", nargs=3, action="append", metavar=("AD", "HIZ", "GEOMETRI"), required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--block", type=int, default=12); ap.add_argument("--stdmax", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=15); ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=100.0)
    ap.add_argument("--yvar", type=float, default=25.0, help="yarıvariogram için |y| alt sınırı (km)")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    prof = profile_from_header(a.profil)
    tr = read_traces(a.faults); offs = fault_offsets(tr, prof, xmax=a.xmax)
    print(f"Profil: merkez {prof.lon0:.4f}D {prof.lat0:.4f}K, doğrultu N{prof.strike:.1f}°D; faylar y = "
          + ", ".join(f"{v:+.2f}" for v in offs.values()) + " km")
    k = a.block
    for name, vpath, gpath in a.track:
        d = read_mintpy(vpath, gpath)
        valid0 = np.isfinite(d["los"]).sum()
        los = np.where(d["std"] > a.stdmax, np.nan, d["los"])
        n_std = valid0 - np.isfinite(los).sum()
        L, npx = block_median(los, k, min_frac=0.5)
        lon, _ = block_median(d["lon"], k, min_frac=0); lat, _ = block_median(d["lat"], k, min_frac=0)
        inc, _ = block_median(np.where(np.isfinite(los), d["incidence"], np.nan), k, 0.5)
        az, _ = block_median(np.where(np.isfinite(los), d["azimuth"], np.nan), k, 0.5)
        anom = local_anomaly_mask(L, window=a.window, thresh=a.thresh, dilate=1)
        x, y = prof.to_profile(lon, lat)
        box = (np.abs(x) <= a.xmax) & (np.abs(y) <= a.ymax)
        good = box & np.isfinite(L) & ~anom
        print(f"\n=== {name} ({d['orbit']}): piksel {valid0:,}; σ>{a.stdmax:g} mm/yıl atılan {n_std:,} piksel")
        print(f"    {k}x{k} blok (~{k*0.08:.2f} km): kutuda {box.sum():,} blok; veri olan {(box & np.isfinite(L)).sum():,};"
              f" yerel anomali (|Δ|>{a.thresh:g} mm/yıl, {a.window} km pencere, 1 blok tampon) {(box & np.isfinite(L) & anom).sum():,}; kalan {good.sum():,}")
        p = los_unit_vector_mintpy(inc[good], az[good])
        out = f"{a.outdir}/insar_{name}_profil.csv"
        np.savetxt(out, np.c_[lon[good], lat[good], x[good], y[good], L[good], npx[good], p[0], p[1], p[2]],
                   fmt="%.5f,%.5f,%.3f,%.3f,%.3f,%d,%.4f,%.4f,%.4f",
                   header=f"profil merkez_lon={prof.lon0:.6f} merkez_lat={prof.lat0:.6f} strike={prof.strike:.4f} iz={name} yorunge={d['orbit']}\n"
                          "lon,lat,x_km,y_km,los_mmyr,npix,pE,pN,pU")
        print(f"    yazıldı: {out}  (LOS medyanı {np.median(L[good]):+.2f}, %2–98 aralığı {np.percentile(L[good], 2):+.2f} … {np.percentile(L[good], 98):+.2f} mm/yıl)")

        # yarıvariogram: fay merkezinden uzak bloklar, düzlem çıkarılmış
        far = good & (np.abs(y) >= a.yvar)
        A = np.c_[np.ones(far.sum()), x[far], y[far]]; c, *_ = np.linalg.lstsq(A, L[far], rcond=None)
        z = L[far] - A @ c
        h, gam, npair = empirical_semivariogram(x[far], y[far], z, bins=np.arange(0, 62, 2.0), max_pairs=400000)
        par = fit_exponential(h, gam, npair)
        print(f"    yarıvariogram (|y|≥{a.yvar:g} km, {far.sum():,} blok, düzlem çıkarılmış): σ = {par['sigma']:.2f} mm/yıl,"
              f" λ = {par['lambda']:.1f} km, nugget = {par['nugget']:.2f} mm/yıl")

        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
        show = np.where(box & np.isfinite(L), L, np.nan)
        vm = np.nanpercentile(np.abs(L[good] - np.median(L[good])), 98)
        sc = ax[0].scatter(x[box & np.isfinite(L) & ~anom], y[box & np.isfinite(L) & ~anom], c=L[box & np.isfinite(L) & ~anom], s=3,
                           cmap="RdBu_r", vmin=np.median(L[good]) - vm, vmax=np.median(L[good]) + vm)
        ax[0].scatter(x[box & np.isfinite(L) & anom], y[box & np.isfinite(L) & anom], s=3, color="0.6", label="maskelenen")
        for parts in tr.values():
            for cc in parts:
                fx, fy = prof.to_profile(cc[:, 0], cc[:, 1]); ax[0].plot(fx, fy, "k-", lw=0.8)
        plt.colorbar(sc, ax=ax[0], label="LOS mm/yıl (+ uyduya doğru)"); ax[0].set_aspect("equal")
        ax[0].set_xlabel("x (km, doğrultu boyunca)"); ax[0].set_ylabel("y (km, + kuzey)"); ax[0].legend(fontsize=7, loc="lower right")
        ax[0].set_title(f"{name}: ~1 km bloklar, profil kutusu")
        ax[1].scatter(y[good], L[good], s=2, alpha=0.3, color="#1f6feb")
        bins = np.arange(-a.ymax, a.ymax + 1, 2.0); mid = 0.5 * (bins[1:] + bins[:-1])
        med = [np.median(L[good][(y[good] >= b0) & (y[good] < b1)]) if np.any((y[good] >= b0) & (y[good] < b1)) else np.nan for b0, b1 in zip(bins[:-1], bins[1:])]
        ax[1].plot(mid, med, "k-", lw=1.5, label="2 km bin medyanı")
        for v in offs.values(): ax[1].axvline(v, ls="--", color="0.5", lw=0.8)
        ax[1].set_xlabel("y (km)"); ax[1].set_ylabel("LOS mm/yıl"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
        ax[1].set_title(f"{name}: profil (rampa MintPy'de çıkarılmış)")
        hh = np.linspace(0, h.max(), 200)
        ax[2].plot(h, gam, "o", ms=4, color="#1f6feb"); ax[2].plot(hh, par["nugget"] ** 2 + par["sigma"] ** 2 * (1 - np.exp(-hh / par["lambda"])), "k-",
                                                                  label=f"σ={par['sigma']:.2f}, λ={par['lambda']:.0f} km, nug={par['nugget']:.2f}")
        ax[2].set_xlabel("mesafe (km)"); ax[2].set_ylabel("γ (mm/yıl)²"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
        ax[2].set_title(f"{name}: ön yarıvariogram (|y|≥{a.yvar:g} km)")
        fig.tight_layout(); fig.savefig(f"{a.outdir}/insar_{name}_ozet.png", dpi=110); print(f"    figür: {a.outdir}/insar_{name}_ozet.png")


if __name__ == "__main__":
    main()
