#!/usr/bin/env python3
"""C — 3B segment modeli ile ortak ters çözüm (profil2d.model3d): GPS + (A) düşeyden arındırılmış yatay InSAR.

Kullanım:
  python scripts/ters_coz_3b.py --gps local/gps_profil.csv --insar local/insar_asc_profil.csv \
      --insar local/insar_dsc_profil.csv --faults FAYLAR.geojson --kurgu blok --uzanti 100 \
      --Dmin 10 --Dmax 15 --out local/ters3b_blok  [--sadece-gps] [--adim 5]
--kurgu derin     : yalnız haritalı iz boyunca D altında kayma (uzanti kullanılmaz)
--kurgu blok      : iz + uzantı boyunca D altında kayma  (= blok hareketi − 0..D geri kayma)
--kurgu blok_krip : blok hareketi − yalnız haritalı izde 0..D geri kayma (uzantılar yüzeyde kripli)
Veri kutusu, GPS seçimi, InSAR hücreleme ve gürültü modeli 2B ters_coz.py ile AYNIDIR.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from ters_coz import header, load_insar, yatay_dataset, summarize
from profil2d.faults import read_traces
from profil2d.inversion import GPSSet, InSARSet, exp_cov, sample
from profil2d.insar import empirical_semivariogram, fit_exponential
from profil2d.model3d import merged_trace, trace_pieces, make_green, JointModel3D

KISA = {"HAVRAN - BALYA FAYI": "HB", "BALIKESİR FAYI": "BF"}


def geometry(path, prof, fault_order, step, ext, kurgu):
    traces = read_traces(path)
    geom, info = [], {}
    for name in fault_order:
        parts = [np.c_[prof.to_profile(c[:, 0], c[:, 1])] for c in traces[name]]
        xn, yn = merged_trace(parts, step=step)
        tr, ex = trace_pieces(xn, yn, ext=0.0 if kurgu == "derin" else ext)
        V = np.vstack(parts)
        dev = np.abs(V[:, 1] - np.interp(V[:, 0], xn, yn))
        info[name] = dict(x=[float(xn[0]), float(xn[-1])], n_parca=len(tr), iz_uzunlugu=float(sum(p["length"] for p in tr)),
                          sapma_medyan=float(np.median(dev)), sapma_max=float(dev.max()), xn=xn.tolist(), yn=yn.tolist(),
                          uzanti=[(p["x0"], p["y0"], p["x1"], p["y1"]) for p in ex])
        geom.append((tr, ex))
    return geom, info, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True); ap.add_argument("--insar", action="append", default=[])
    ap.add_argument("--faults", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kurgu", choices=["derin", "blok", "blok_krip"], required=True)
    ap.add_argument("--uzanti", type=float, default=100.0); ap.add_argument("--adim", type=float, default=5.0)
    ap.add_argument("--Dmin", type=float, default=10.0); ap.add_argument("--Dmax", type=float, default=15.0)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=50.0)
    ap.add_argument("--cell", type=float, default=2.0); ap.add_argument("--sadece-gps", action="store_true")
    ap.add_argument("--nwalkers", type=int, default=32); ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    prof, fy = header(a.gps)
    names = list(fy.keys())

    # ---- geometri ----
    geom, ginfo, traces = geometry(a.faults, prof, names, a.adim, a.uzanti, a.kurgu)
    print(f"Profil N{prof.strike:.1f}°D; kurgu '{a.kurgu}'" + ("" if a.kurgu == "derin" else f", uzantı {a.uzanti:g} km"))
    for nm in names:
        g = ginfo[nm]
        print(f"  {KISA.get(nm, nm)}: birleşik iz x [{g['x'][0]:.1f}, {g['x'][1]:.1f}] km, {g['n_parca']} parça, "
              f"{g['iz_uzunlugu']:.1f} km; köşelerin ize uzaklığı medyan {g['sapma_medyan']:.2f}, en çok {g['sapma_max']:.2f} km")

    # ---- veri (2B ile aynı seçim) ----
    gg = np.genfromtxt(a.gps, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,
                       names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
    kg = (np.abs(gg["x"]) <= a.xmax) & (np.abs(gg["y"]) <= a.ymax)
    gps = GPSSet(y=gg["y"][kg], x=gg["x"][kg], vpar=gg["vpar"][kg], vperp=gg["vperp"][kg], spar=gg["spar"][kg], sperp=gg["sperp"][kg])
    print(f"GPS {kg.sum()} istasyon (|x|≤{a.xmax:g}, |y|≤{a.ymax:g} km)")
    tracks, covinfo = [], {}
    if not a.sadece_gps:
        raw = []
        for path in a.insar:
            name, x, y, los, pE, pN, pU, _ = load_insar(path, a.xmax, a.ymax, a.cell)
            raw.append(InSARSet(name=name, x=x, y=y, los=los, pE=pE, pN=pN, pU=pU, cov0=np.eye(len(x))))
        ds, par, kap = yatay_dataset(raw, prof, a.cell, a.ymax); tracks = [ds]; covinfo["yatay"] = par
        print(f"InSAR (A) yatay: {len(ds.los):,} hücre; κ medyan {np.median(kap):+.3f}; C0 σ={par['sigma']:.2f}, λ={par['lambda']:.1f} km, nugget={par['nugget']:.2f}")
    xo = np.concatenate([gps.x] + [t.x for t in tracks]); yo = np.concatenate([gps.y] + [t.y for t in tracks])

    t0 = time.time()
    model = JointModel3D(prof.s, prof.n, gps, tracks, green=make_green(xo, yo, geom, a.kurgu), fault_names=names,
                         bounds=dict(D=(a.Dmin, a.Dmax)))
    print(f"Green fonksiyonları ({len(model._spl.grid)} D değeri, {len(xo):,} nokta) {time.time() - t0:.1f} sn; "
          f"doğrusal olmayan {model.theta_names}; doğrusal {len(model.lin_names)}")
    t0 = time.time()
    out = sample(model, nwalkers=a.nwalkers, nsteps=a.nsteps, burn=a.burn, seed=a.seed, ndraw_lin=4000)
    lmax = float(model.log_post(out["start"]))
    print(f"Örnekleme {time.time() - t0:.0f} sn; kabul {np.mean(out['acc']):.2f}; τ = "
          + ", ".join(f"{n}:{t:.0f}" for n, t in zip(model.theta_names, out["tau"])) + f" (zincir {a.nsteps - a.burn}); en büyük log p(d|θ) = {lmax:.2f}")

    th = out["theta"].copy(); tn = list(model.theta_names)
    for j, nm in enumerate(tn):
        if nm.startswith("log_"): th[:, j] = np.exp(th[:, j]); tn[j] = nm[4:]
    S_th, S_lin = summarize(th, tn, "theta"), summarize(out["lin"], model.lin_names, "lin")
    V = out["lin"][:, :model.K]; Vs = V.sum(1)
    q = lambda v: [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print("\nPosterior (medyan [%2.5–97.5]):")
    for nm in tn: print(f"  {nm:16s} {S_th[nm]['median']:8.2f}  [{S_th[nm]['lo']:7.2f} … {S_th[nm]['hi']:7.2f}]")
    for k, nm in enumerate(names):
        v = q(V[:, k]); print(f"  V_{KISA.get(nm, nm):14s} {v[0]:8.2f}  [{v[1]:7.2f} … {v[2]:7.2f}]   P(V>0) = {np.mean(V[:, k] > 0):.2f}")
    v = q(Vs); print(f"  V_toplam         {v[0]:8.2f}  [{v[1]:7.2f} … {v[2]:7.2f}]   korelasyon V_1–V_2 = {np.corrcoef(V[:, 0], V[:, 1])[0, 1]:+.2f}")
    for nm in ("a_par", "a_perp", "b_perp"):
        print(f"  {nm:16s} {S_lin[nm]['median']:8.3f}  [{S_lin[nm]['lo']:7.3f} … {S_lin[nm]['hi']:7.3f}]")

    th_med = np.median(out["theta"], axis=0); m_med, _ = model.conditional(th_med)
    pred = model.predict(th_med, m_med)
    rms = dict(gps_par=float(np.sqrt(np.mean((gps.vpar - pred["gps_par"]) ** 2))), gps_perp=float(np.sqrt(np.mean((gps.vperp - pred["gps_perp"]) ** 2))))
    for t in tracks: rms[t.name] = float(np.sqrt(np.mean((t.los - pred[t.name]) ** 2)))
    print("  rezidüel RMS (mm/yıl): " + ", ".join(f"{k} {v:.2f}" for k, v in rms.items()))
    res = dict(kurgu=a.kurgu, uzanti=None if a.kurgu == "derin" else a.uzanti, sadece_gps=a.sadece_gps, D_onsel=[a.Dmin, a.Dmax],
               faylar=names, theta=S_th, lin=S_lin, V={KISA.get(nm, nm): q(V[:, k]) for k, nm in enumerate(names)}, V_toplam=q(Vs),
               P_Vpos={KISA.get(nm, nm): float(np.mean(V[:, k] > 0)) for k, nm in enumerate(names)},
               korelasyon_V=float(np.corrcoef(V[:, 0], V[:, 1])[0, 1]), log_marjinal_max=lmax, rms=rms, cov=covinfo,
               acc=float(np.mean(out["acc"])), tau=[float(t) for t in out["tau"]], geometri=ginfo,
               n_gps=int(kg.sum()), n_insar={t.name: len(t.los) for t in tracks})
    json.dump(res, open(f"{a.out}/ozet.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"{a.out}/posterior.npz", theta=out["theta"], lin=out["lin"], logp=out["logp"], chain=out["chain"],
                        theta_names=np.array(model.theta_names), lin_names=np.array(model.lin_names))

    # ---- figürler ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    D_med = float(th_med[0]); Vm = m_med[:model.K]
    gx, gy = np.meshgrid(np.linspace(-a.xmax, a.xmax, 81), np.linspace(-a.ymax, a.ymax, 81))
    Ux, Uy, _ = make_green(gx.ravel(), gy.ravel(), geom, a.kurgu)(D_med)
    vpar_map = (Ux @ Vm).reshape(gx.shape); vperp_map = (Uy @ Vm).reshape(gx.shape)
    ia, ip, ib = (model.lin_names.index(k) for k in ("a_par", "a_perp", "b_perp"))
    obs_par = gps.vpar - m_med[ia]; obs_perp = gps.vperp - m_med[ip] - m_med[ib] * gps.y
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
    lim = np.abs(vpar_map).max()
    for j, (fld, ttl) in enumerate(((vpar_map, "model faya paralel hız"), (vperp_map, "model faya dik hız"))):
        im = ax[j].pcolormesh(gx, gy, fld, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        plt.colorbar(im, ax=ax[j], shrink=.8, label="mm/yıl")
        for nm in names:
            for c in traces[nm]:
                x_, y_ = prof.to_profile(c[:, 0], c[:, 1]); ax[j].plot(x_, y_, color="0.3", lw=.6)
            g = ginfo[nm]; ax[j].plot(g["xn"], g["yn"], "k-", lw=1.6)
            for (x0, y0, x1, y1) in g["uzanti"]:
                ax[j].plot([x0, x1], [y0, y1], "k--", lw=1)
        ax[j].set_xlim(-a.xmax, a.xmax); ax[j].set_ylim(-a.ymax, a.ymax); ax[j].set_aspect("equal")
        ax[j].set_title(ttl); ax[j].set_xlabel("x (km, doğrultu boyunca)"); ax[j].set_ylabel("y (km, + kuzey)")
    sc = 6.0
    ax[0].quiver(gps.x, gps.y, obs_par, obs_perp, color="#1f6feb", scale=sc, scale_units="inches", width=.004, label="GPS − nuisance")
    ax[0].quiver(gps.x, gps.y, pred["gps_par"] - m_med[ia], pred["gps_perp"] - m_med[ip] - m_med[ib] * gps.y, color="#cf222e",
                 scale=sc, scale_units="inches", width=.003, label="model (fay)")
    ax[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"3B '{a.kurgu}'" + ("" if a.kurgu == "derin" else f" (uzantı {a.uzanti:g} km)")
                 + f": D={D_med:.1f} km, " + ", ".join(f"V_{KISA.get(nm, nm)}={Vm[k]:.2f}" for k, nm in enumerate(names)) + " mm/yıl")
    fig.tight_layout(); fig.savefig(f"{a.out}/harita.png", dpi=100); plt.close(fig)

    # dik kesitler (x = −30, 0, +30; ±15 km bant)
    pick = np.random.default_rng(1).choice(len(out["theta"]), min(200, len(out["theta"])), replace=False)
    yy = np.linspace(-a.ymax, a.ymax, 101)
    fig, ax = plt.subplots(2 if tracks else 1, 3, figsize=(16, 8 if tracks else 4.5), squeeze=False, sharey="row")
    for c_, x0 in enumerate((-30.0, 0.0, 30.0)):
        Gs = make_green(np.full_like(yy, x0), yy, geom, a.kurgu)
        cache = {}
        curves = []
        for i in pick:
            Dk = round(float(out["theta"][i, 0]), 1)
            if Dk not in cache: cache[Dk] = Gs(Dk)
            curves.append(cache[Dk][0] @ out["lin"][i, :model.K])
        curves = np.array(curves)
        b = np.abs(gps.x - x0) <= 15
        ax[0, c_].errorbar(gps.y[b], obs_par[b], yerr=np.sqrt(gps.spar[b] ** 2 + np.exp(th_med[1]) ** 2), fmt="o", ms=4, color="#1f6feb", label="GPS paralel − a_par")
        ax[0, c_].fill_between(yy, *np.percentile(curves, [2.5, 97.5], 0), color="#1a7f37", alpha=.25, label="3B model %95")
        ax[0, c_].plot(yy, np.median(curves, 0), color="#1a7f37", lw=2)
        ax[0, c_].set_title(f"x = {x0:+.0f} km (±15 km bant)")
        if tracks:
            tr = tracks[0]; bb = np.abs(tr.x - x0) <= 15
            jj = model.K + 3
            h = tr.los - (m_med[jj] + m_med[jj + 1] * tr.x + m_med[jj + 2] * tr.y) - (tr.pE * prof.n[0] + tr.pN * prof.n[1]) * m_med[ib] * tr.y
            bins = np.arange(-a.ymax, a.ymax + .1, 4.0); mid = .5 * (bins[1:] + bins[:-1])
            med = [np.median(h[bb & (tr.y >= l) & (tr.y < u)]) if np.any(bb & (tr.y >= l) & (tr.y < u)) else np.nan for l, u in zip(bins[:-1], bins[1:])]
            ax[1, c_].plot(tr.y[bb], h[bb], ".", ms=2, color="0.6"); ax[1, c_].plot(mid, med, "o-", color="#bf8700", ms=4, label="yatay − rampa − dik terim (4 km medyan)")
            kap = np.median(tr.pE * prof.n[0] + tr.pN * prof.n[1])
            Ux_s, Uy_s, _ = Gs(D_med)
            ax[1, c_].plot(yy, Ux_s @ Vm + kap * (Uy_s @ Vm), color="#1a7f37", lw=2, label="model (q·u, medyan κ)")
            ax[1, c_].set_ylim(-8, 8)
        for r_ in ax[:, c_]:
            r_.grid(alpha=.3)
            for nm in names: r_.axvline(np.interp(x0, ginfo[nm]["xn"], ginfo[nm]["yn"], left=np.nan, right=np.nan), ls="--", color="0.4", lw=.8)
        ax[-1, c_].set_xlabel("y (km, + kuzey)")
    ax[0, 0].set_ylabel("mm/yıl"); ax[0, 0].legend(fontsize=7)
    if tracks: ax[1, 0].set_ylabel("mm/yıl"); ax[1, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.out}/kesitler.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].scatter(V[:, 0], V[:, 1], s=2, alpha=.3); ax[0].set_xlabel(f"V_{KISA.get(names[0])} (mm/yıl)"); ax[0].set_ylabel(f"V_{KISA.get(names[1])} (mm/yıl)")
    ax[1].hist(Vs, bins=50, color="#1a7f37", alpha=.7); ax[1].set_xlabel("V toplam (mm/yıl, + sağ-yanal)")
    ax[2].hist(out["theta"][:, 0], bins=40, color="#8250df", alpha=.7); ax[2].set_xlabel("D (km)")
    for x_ in ax: x_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/posterior.png", dpi=100); plt.close(fig)
    print(f"\nçıktılar: {a.out}/ozet.json, posterior.npz, harita.png, kesitler.png, posterior.png")


if __name__ == "__main__":
    main()
