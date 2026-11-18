#!/usr/bin/env python3
"""GPS faya dik bileşeninde doğrultu boyunca gradyan (dv_perp/dx) var mı, sağlam mı?

v_perp = a + Lyx·x + b·y eşit ağırlıklı EKK ile alt kümelerde (uçlar hariç, kuzey/güney blok) ve tek istasyon
çıkarılarak (jackknife) kestirilir; x bölmelerinde medyanlar ve bir figür üretir.
Kullanım: python scripts/dik_gradyan_testi.py --gps local/gps_profil.csv --out local/dik_gradyan.png [--xmax 50 --ymax 50]
"""
import argparse, os, sys
import numpy as np


def fit(x, y, v):
    A = np.c_[np.ones(len(x)), x, y]; c, *_ = np.linalg.lstsq(A, v, rcond=None)
    r = v - A @ c; cov = np.linalg.inv(A.T @ A) * r.var(ddof=3)
    return c, np.sqrt(np.diag(cov)), r.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=50.0)
    a = ap.parse_args()
    g = np.genfromtxt(a.gps, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,
                      names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
    k = (np.abs(g["x"]) <= a.xmax) & (np.abs(g["y"]) <= a.ymax)
    x, y, vn = g["x"][k], g["y"][k], g["vperp"][k]
    subsets = [("tümü", np.ones(len(x), bool)), (f"|x|≤{0.7 * a.xmax:g}", np.abs(x) <= 0.7 * a.xmax),
               (f"x≤{0.6 * a.xmax:g} (doğu ucu hariç)", x <= 0.6 * a.xmax), (f"x≥−{0.6 * a.xmax:g} (batı ucu hariç)", x >= -0.6 * a.xmax),
               ("y≥0 (kuzey)", y >= 0), ("y<0 (güney)", y < 0)]
    print("v_perp = a + Lyx·x + b·y (EKK, eşit ağırlık):")
    for nm, s in subsets:
        c, sd, rs = fit(x[s], y[s], vn[s])
        print(f"  {nm:26s} n={s.sum():2d}  Lyx = {c[1]:+.3f} ± {sd[1]:.3f} mm/yıl/km   b = {c[2]:+.3f}   rezidüel sd {rs:.2f}")
    jk = np.array([fit(np.delete(x, i), np.delete(y, i), np.delete(vn, i))[0][1] for i in range(len(x))])
    print(f"  tek istasyon çıkarma: Lyx ∈ [{jk.min():+.3f}, {jk.max():+.3f}]")
    c, sd, _ = fit(x, y, vn)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = vn - c[0] - c[2] * y
    sc = ax.scatter(x, r, c=y, cmap="coolwarm", s=30, edgecolor="k", lw=.4)
    xx = np.linspace(-a.xmax, a.xmax, 2); ax.plot(xx, c[1] * xx, "k-", label=f"Lyx = {c[1]:+.3f} ± {sd[1]:.3f} mm/yıl/km")
    plt.colorbar(sc, label="y (km)"); ax.set_xlabel("x (km, doğrultu boyunca)"); ax.set_ylabel("v_perp − a − b·y (mm/yıl)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print(f"figür: {a.out}")


if __name__ == "__main__":
    main()
