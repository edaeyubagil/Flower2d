#!/usr/bin/env python3
"""GPS verisini 2B profil modeline hazırlar (Aşama 1–2).

Kullanım:
  python scripts/hazirla_gps.py --org VEL.org --faults FAYLAR.geojson --haric local/gps_haric.txt \
         --out local/gps_profil.csv [--xmax 50] [--ymax 100] [--fig local/gps_profil.png]

* Profil merkezi = fay köşelerinin ortalaması; doğrultu = segment azimutlarının uzunlukla ağırlıklı
  eksenel ortalaması (profil2d.faults.principal_strike).
* --haric dosyasındaki istasyonlar atılır (satır başına: site [neden]; '#' yorum).
* Formal kovaryans (sE, sN, rho) faya paralel/dik eksenlere döndürülür. Gerçekçi hata ayrıca
  inversiyonda kestirilir (Aşama 3); burada formal değerler olduğu gibi yazılır.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profil2d.geometry import Profile
from profil2d.faults import read_traces, principal_strike, centroid, fault_offsets
from profil2d.gps import read_globk_org
from profil2d.forward import Fault, fault_parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True); ap.add_argument("--faults", required=True)
    ap.add_argument("--haric", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=100.0)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()

    tr = read_traces(a.faults)
    lon0, lat0 = centroid(tr); strike = principal_strike(tr)
    prof = Profile(lon0, lat0, strike)
    offs = fault_offsets(tr, prof, xmax=a.xmax)
    print(f"Profil: merkez {lon0:.4f}D {lat0:.4f}K, doğrultu N{strike:.1f}°D (y+ = sol = N{(strike - 90) % 360:.0f}°D yönü)")
    for k, v in offs.items(): print(f"  fay '{k}': y = {v:+.2f} km")

    g = read_globk_org(a.org)
    excl = {l.split()[0] for l in open(a.haric) if l.strip() and not l.startswith("#")}
    missing = excl - set(g["site"])
    if missing: print("UYARI: dışlama listesinde olup veride olmayan:", sorted(missing))
    x, y = prof.to_profile(g["lon"], g["lat"])
    box = (np.abs(x) <= a.xmax) & (np.abs(y) <= a.ymax)
    keep = box & ~np.isin(g["site"], list(excl))
    print(f"Kutu |x|<={a.xmax:g}, |y|<={a.ymax:g} km: {box.sum()} istasyon; dışlanan {(box & ~keep).sum()}; kalan {keep.sum()}")

    vpar, vperp = prof.enu_to_profile(g["ve"], g["vn"])
    R = np.array([prof.s, prof.n])
    spar, sperp, rpp = (np.zeros(len(x)) for _ in range(3))
    for i in range(len(x)):
        c = g["rho"][i] * g["se"][i] * g["sn"][i]
        C = R @ np.array([[g["se"][i] ** 2, c], [c, g["sn"][i] ** 2]]) @ R.T
        spar[i], sperp[i] = np.sqrt(C[0, 0]), np.sqrt(C[1, 1]); rpp[i] = C[0, 1] / (spar[i] * sperp[i])
    hdr = (f"profil merkez_lon={lon0:.6f} merkez_lat={lat0:.6f} strike={strike:.4f} "
           + " ".join(f"fay[{k}]={v:.4f}" for k, v in offs.items())
           + "\nsite,lon,lat,x_km,y_km,ve,vn,se,sn,rho,vpar,vperp,spar,sperp,rho_pp")
    with open(a.out, "w") as f:
        f.write("# " + hdr.replace("\n", "\n") + "\n")
        for i in np.flatnonzero(keep):
            f.write(f"{g['site'][i]},{g['lon'][i]:.5f},{g['lat'][i]:.5f},{x[i]:.3f},{y[i]:.3f},{g['ve'][i]:.2f},{g['vn'][i]:.2f},"
                    f"{g['se'][i]:.2f},{g['sn'][i]:.2f},{g['rho'][i]:.3f},{vpar[i]:.3f},{vperp[i]:.3f},{spar[i]:.3f},{sperp[i]:.3f},{rpp[i]:.3f}\n")
    print("yazıldı:", a.out)

    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = keep; yy = np.linspace(-a.ymax, a.ymax, 400)
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].errorbar(y[k], vpar[k], yerr=np.maximum(spar[k], 0.05), fmt="o", ms=4, color="#1f6feb", label=f"GPS (n={k.sum()}), formal σ")
        base = np.median(vpar[k])
        for D, c in [(5, "#8250df"), (15, "#1a7f37"), (30, "#bf8700")]:
            m = base + fault_parallel(yy, [Fault(y0=v, D=D, V=1.0) for v in offs.values()])
            ax[0].plot(yy, m, color=c, lw=1.5, label=f"örnek: her fayda V=1 mm/yıl, D={D} km (uyum DEĞİL)")
        for kname, v in offs.items():
            for a_ in ax: a_.axvline(v, ls="--", color="0.5", lw=0.8)
            ax[0].text(v, ax[0].get_ylim()[1], kname.title(), rotation=90, va="top", fontsize=7)
        ax[0].set_ylabel(f"faya paralel (N{strike:.0f}°D) mm/yıl"); ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=.3)
        ax[1].errorbar(y[k], vperp[k], yerr=np.maximum(sperp[k], 0.05), fmt="o", ms=4, color="#1a7f37")
        ax[1].set_ylabel(f"faya dik (N{(strike - 90) % 360:.0f}°D) mm/yıl"); ax[1].set_xlabel("y (km, + = kuzey tarafı)"); ax[1].grid(alpha=.3)
        fig.suptitle("Temizlenmiş GPS profili (Avrasya sabit) — V>0 sağ-yanal"); fig.tight_layout(); fig.savefig(a.fig, dpi=130)
        print("figür:", a.fig)


if __name__ == "__main__":
    main()
