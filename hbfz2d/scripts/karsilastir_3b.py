#!/usr/bin/env python3
"""3B kurguların karşılaştırması: (1) V_toplam posteriorları (+ isteğe bağlı 2B çözüm), (2) GPS-yalnız modellerde
kurgu tercihini hangi bileşenin sürdüğü (paralel / dik / ikisi; kapatılan bileşenin formal σ'sı 1e3 yapılır,
böylece bilgi taşımaz ve sabit terimleri kurgulardan bağımsız kalır).

Kullanım:
  python scripts/karsilastir_3b.py --gps local/gps_profil.csv --faults FAYLAR.geojson \
      --kosu derin=local/ters3b_derin --kosu "blok 100 km=local/ters3b_blok100" ... [--iki-b local/ters_B_A_tek] --out local/ters3b_karsilastirma
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from ters_coz import header
from ters_coz_3b import geometry
from profil2d.inversion import GPSSet
from profil2d.model3d import make_green, JointModel3D


def bilesen_testi(gps_path, faults, Dmin, Dmax, xmax, ymax):
    prof, fy = header(gps_path); names = list(fy)
    g = np.genfromtxt(gps_path, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,
                      names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
    k = (np.abs(g["x"]) <= xmax) & (np.abs(g["y"]) <= ymax)
    out = {}
    for label, kurgu, ext in (("derin", "derin", 0.0), ("blok 100 km", "blok", 100.0), ("blok ∞", "blok", 1e6), ("blok_krip 100 km", "blok_krip", 100.0)):
        geom, _, _ = geometry(faults, prof, names, 5.0, ext, kurgu)
        out[label] = {}
        for bil in ("ikisi", "paralel", "dik"):
            spar = g["spar"][k] if bil != "dik" else np.full(k.sum(), 1e3)
            sperp = g["sperp"][k] if bil != "paralel" else np.full(k.sum(), 1e3)
            gps = GPSSet(y=g["y"][k], x=g["x"][k], vpar=g["vpar"][k], vperp=g["vperp"][k], spar=spar, sperp=sperp)
            m = JointModel3D(prof.s, prof.n, gps, [], green=make_green(gps.x, gps.y, geom, kurgu), fault_names=names, bounds=dict(D=(Dmin, Dmax)))
            f = lambda t: -m.log_post(t) if np.isfinite(m.log_post(t)) else 1e30
            best = min((minimize(f, [D0, np.log(.7), np.log(1.9)], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
                        for D0 in np.linspace(Dmin + .1 * (Dmax - Dmin), Dmax - .1 * (Dmax - Dmin), 3)), key=lambda r: r.fun)
            mu, cov = m.conditional(best.x)
            out[label][bil] = dict(logp=float(-best.fun), V_toplam=float(mu[0] + mu[1]), sd=float(np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])), D=float(best.x[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True); ap.add_argument("--faults", required=True)
    ap.add_argument("--kosu", action="append", required=True, help="etiket=dizin (ters_coz_3b.py çıktısı)")
    ap.add_argument("--iki-b", help="2B tek-fay çözüm dizini (ters_coz.py çıktısı), V_1 karşılaştırma için")
    ap.add_argument("--Dmin", type=float, default=10.0); ap.add_argument("--Dmax", type=float, default=15.0)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5)); bins = np.linspace(-1, 16, 120)
    ozet = {}
    for item in a.kosu:
        label, d = item.split("=", 1)
        p = np.load(f"{d}/posterior.npz"); ln = list(p["lin_names"])
        vs = p["lin"][:, ln.index("V_1")] + p["lin"][:, ln.index("V_2")]
        r = json.load(open(f"{d}/ozet.json"))
        ozet[label] = dict(V_toplam=r["V_toplam"], V=r["V"], korelasyon_V=r["korelasyon_V"], log_marjinal_max=r["log_marjinal_max"])
        ax.hist(vs, bins=bins, histtype="step", lw=1.8, density=True, label=f"3B {label}: {np.median(vs):.2f} [{np.percentile(vs, 2.5):.2f}, {np.percentile(vs, 97.5):.2f}]")
    if a.iki_b:
        p = np.load(f"{a.iki_b}/posterior.npz"); v = p["lin"][:, list(p["lin_names"]).index("V_1")]
        ozet["2B tek fay"] = dict(V_toplam=[float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=.25, color="0.3", density=True, label=f"2B tek fay: {np.median(v):.2f} [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
    ax.set_xlabel("V toplam = V_HB + V_BF (mm/yıl, + sağ-yanal)"); ax.set_ylabel("olasılık yoğunluğu"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title(f"GPS + yatay InSAR, D ~ U[{a.Dmin:g}, {a.Dmax:g}] km")
    fig.tight_layout(); fig.savefig(f"{a.out}/V_toplam.png", dpi=110)

    bt = bilesen_testi(a.gps, a.faults, a.Dmin, a.Dmax, a.xmax, a.ymax)
    print("GPS-yalnız, en büyük log p(d|θ) ve o noktadaki V_toplam (koşullu ortalama ± sd):")
    for label, r in bt.items():
        print(f"  {label:17s} " + " | ".join(f"{b}: {r[b]['logp']:8.2f}  V {r[b]['V_toplam']:6.2f}±{r[b]['sd']:.2f}" for b in ("ikisi", "paralel", "dik")))
    json.dump(dict(kosular=ozet, bilesen_testi=bt), open(f"{a.out}/ozet.json", "w"), ensure_ascii=False, indent=1)
    print(f"çıktılar: {a.out}/V_toplam.png, ozet.json")


if __name__ == "__main__":
    main()
