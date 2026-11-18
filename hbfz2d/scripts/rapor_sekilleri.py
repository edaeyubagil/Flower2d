#!/usr/bin/env python3
"""Rapor için açıklayıcı şekiller: (1) çalışma alanı, (2) 3B kurguların birim hızda davranışı, (3) doğrulama örnekleri.

Kullanım: python scripts/rapor_sekilleri.py --gps local/gps_profil.csv --insar local/insar_asc_profil.csv \
              --faults FAYLAR.geojson --out local/rapor
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from ters_coz import header
from ters_coz_3b import geometry, KISA
from profil2d.forward import screw
from profil2d.okada import okada_rect
from profil2d.model3d import make_green

RENK = {"HB": "#1f6feb", "BF": "#cf222e"}


def alan(a, prof, names, plt):
    g = np.genfromtxt(a.gps, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,
                      names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
    ins = np.loadtxt(a.insar, delimiter=",", comments="#")
    seg = {}
    for f in json.load(open(a.faults))["features"]:
        p = f["properties"]; c = np.asarray(f["geometry"]["coordinates"], float)[:, :2]
        seg.setdefault((p["FAY_ADI"], p["SEGMENT_NO"], p["SEGMENT_ADI"]), []).append(c)
    _, info, _ = geometry(a.faults, prof, names, 5.0, 100.0, "blok")
    fig, ax = plt.subplots(figsize=(10, 7.2))
    k = (np.abs(ins[:, 2]) <= 50) & (np.abs(ins[:, 3]) <= 50)
    ax.scatter(ins[k, 2], ins[k, 3], s=1, color="0.85", label="InSAR 1 km blokları (asc)", zorder=0)
    for (fay, no, ad), parts in sorted(seg.items(), key=lambda kv: kv[0][1]):
        xs, ys = [], []
        for c in parts:
            x, y = prof.to_profile(c[:, 0], c[:, 1]); xs.append(x); ys.append(y)
            ax.plot(x, y, color=RENK[KISA[fay]], lw=.9, alpha=.8)
        X, Y = np.concatenate(xs), np.concatenate(ys)
        ax.annotate(f"{no}\n{ad.replace(' Segmenti', '')}", (X.mean(), Y.mean()), xytext=(0, 9 if KISA[fay] == "HB" else -18),
                    textcoords="offset points", ha="center", fontsize=7, color=RENK[KISA[fay]])
    for nm in names:
        gi = info[nm]
        ax.plot(gi["xn"], gi["yn"], "-", color=RENK[KISA[nm]], lw=2.6, alpha=.55, label=f"{KISA[nm]} birleşik iz ({gi['iz_uzunlugu']:.0f} km)")
        for (x0, y0, x1, y1) in gi["uzanti"]:
            ax.plot([x0, x1], [y0, y1], "--", color=RENK[KISA[nm]], lw=1)
    inb = (np.abs(g["x"]) <= 50) & (np.abs(g["y"]) <= 50)
    ax.plot(g["x"][inb], g["y"][inb], "k^", ms=6, label=f"GPS, kutu içinde ({inb.sum()})")
    ax.plot(g["x"][~inb], g["y"][~inb], "^", mfc="none", mec="k", ms=6, label=f"GPS, kutu dışında ({(~inb).sum()})")
    ax.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50], "k:", lw=1.2, label="çözüm kutusu |x|, |y| ≤ 50 km")
    ax.set_xlim(-62, 62); ax.set_ylim(-65, 100); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel(f"x (km) — doğrultu boyunca, N{prof.strike:.1f}°D"); ax.set_ylabel("y (km) — doğrultuya dik, + kuzey (KKB)")
    ax.legend(fontsize=7.5, loc="upper right"); ax.set_title("Çalışma alanı: segmentler, birleşik izler (kesikli: zon doğrultusunda uzatma) ve GPS")
    fig.tight_layout(); fig.savefig(f"{a.out}/alan_haritasi.png", dpi=110); plt.close(fig)


def kurgular(a, prof, names, plt):
    D = 12.0
    lab = [("derin", "derin", 0.0), ("blok, 100 km uzatma", "blok", 100.0), ("blok, sonsuz uzatma", "blok", 1e6), ("blok_krip, 100 km", "blok_krip", 100.0)]
    col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    _, info, _ = geometry(a.faults, prof, names, 5.0, 0.0, "derin")
    y0s = [float(np.interp(0.0, info[nm]["xn"], info[nm]["yn"])) for nm in names]
    yy = np.linspace(-50, 50, 201); xx = np.linspace(-50, 50, 201)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for (l, kurgu, ext), c in zip(lab, col):
        geom, _, _ = geometry(a.faults, prof, names, 5.0, ext, kurgu)
        Ux = make_green(np.zeros_like(yy), yy, geom, kurgu)(D)[0]
        ax[0].plot(yy, Ux @ np.full(len(names), 0.5), color=c, lw=2, label=l)
        Ux2 = make_green(np.r_[xx, xx], np.r_[np.full_like(xx, -40.0), np.full_like(xx, 40.0)], geom, kurgu)(D)[0] @ np.full(len(names), 0.5)
        ax[1].plot(xx, Ux2[len(xx):] - Ux2[:len(xx)], color=c, lw=2, label=l)
    ax[0].plot(yy, sum(screw(yy, y0, D, 0.5) for y0 in y0s), "k--", lw=1.5, label="2B (sonsuz düz faylar)")
    ax[1].axhline(sum(screw(40.0, y0, D, 0.5) - screw(-40.0, y0, D, 0.5) for y0 in y0s), color="k", ls="--", lw=1.5, label="2B")
    for j, nm in enumerate(names):
        yb = -0.06 - 0.06 * j
        ax[1].plot(info[nm]["x"], [yb, yb], color=RENK[KISA[nm]], lw=5, solid_capstyle="butt")
        ax[1].text(info[nm]["x"][1] + 1.5, yb, f"{KISA[nm]} haritalı uzunluk", va="center", fontsize=7.5, color=RENK[KISA[nm]])
        ax[0].axvline(float(np.interp(0.0, info[nm]["xn"], info[nm]["yn"])), color=RENK[KISA[nm]], ls=":", lw=1)
    ax[1].set_ylim(-0.16, 0.9); ax[1].set_xlim(-62, 75)
    ax[0].set_xlabel("y (km, + kuzey) — x = 0 kesiti"); ax[0].set_ylabel("faya paralel hız (mm/yıl)")
    ax[0].set_title(f"(a) Toplam V = 1 mm/yıl (her fay 0,5), D = {D:g} km")
    ax[1].set_xlabel("x (km, doğrultu boyunca)"); ax[1].set_ylabel("v_par(y=+40) − v_par(y=−40) (mm/yıl)")
    ax[1].set_title("(b) Zonun iki yanı (y = ±40 km) arasındaki hız farkı")
    for x_ in ax: x_.grid(alpha=.3)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(f"{a.out}/kurgu_karsilastirma.png", dpi=110); plt.close(fig)


def dogrulama(a, plt):
    from test_model3d import synthetic3d, _model
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    y = np.linspace(-60, 60, 241)
    for D, c in ((5.0, "#8250df"), (12.0, "#1a7f37"), (25.0, "#bf8700")):
        ue, un, uu = okada_rect(np.zeros_like(y), y, 0.0, 0.0, 90.0, length=1e6, top=D, bottom=1e5, ss=1.0)
        ax[0].plot(y, ue, color=c, lw=3, alpha=.5, label=f"Okada 3B, D={D:g} km")
        ax[0].plot(y, screw(y, 0.0, D, 1.0), "k--", lw=1)
    ax[0].plot([], [], "k--", label="2B vida dislokasyonu (V/π)·atan(y/D)")
    ax[0].set_xlabel("y (km)"); ax[0].set_ylabel("faya paralel hız (V = 1)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title("Uzun düz fayda 3B Okada = 2B çözüm (fark < 3·10⁻⁴)")
    z = []
    for seed in range(10, 30):
        gps, tracks, x, y_ = synthetic3d(V=(2.0, 1.0), D=12.0, variant="blok", seed=seed, gps_sd=0.15, sig=0.5, nug=0.2)
        mean, cov = _model(gps, tracks, x, y_, variant="blok").conditional(np.array([12.0, np.log(.15), np.log(.15), 0.0]))
        z.append((mean[0] + mean[1] - 3.0) / np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]))
    z = np.array(z); zz = np.linspace(-3.5, 3.5, 200)
    ax[1].hist(z, bins=np.arange(-3.5, 3.6, 0.5), density=True, color="#1f6feb", alpha=.6, label=f"20 sentetik veri seti: ort {z.mean():+.2f}, sd {z.std():.2f}")
    ax[1].plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k-", label="beklenen N(0, 1)")
    ax[1].set_xlabel("z = (tahmin − gerçek) / σ   (V_toplam)"); ax[1].set_ylabel("yoğunluk"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_title("Kalibrasyon: belirsizlikler ne iyimser ne kötümser")
    fig.tight_layout(); fig.savefig(f"{a.out}/dogrulama.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True); ap.add_argument("--insar", required=True)
    ap.add_argument("--faults", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    prof, fy = header(a.gps); names = list(fy)
    alan(a, prof, names, plt); kurgular(a, prof, names, plt); dogrulama(a, plt)
    print(f"şekiller: {a.out}/alan_haritasi.png, kurgu_karsilastirma.png, dogrulama.png")


if __name__ == "__main__":
    main()
