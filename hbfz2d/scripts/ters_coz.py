#!/usr/bin/env python3
"""2B ortak ters çözüm: GPS + InSAR izleri (profil2d.inversion).

Kullanım:
  python scripts/ters_coz.py --gps local/gps_profil.csv --insar local/insar_asc_profil.csv \
      --insar local/insar_dsc_profil.csv --model tek --out local/ters_tek [--ymax 50] [--xmax 50] [--cell 2]
      [--nwalkers 32] [--nsteps 4000] [--burn 1500]
--model tek : tek fay, konumu (y0) serbest [-15, 15] km
--model iki : profil başlığındaki iki fay konumunda iki fay (konumlar sabit)
--yatay     : (A) iki izi düşeyi tam eleyen kombinasyonla faya paralel yatay bileşene birleştir; InSAR tek veri seti olur
"""
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from profil2d.geometry import Profile
from profil2d.inversion import GPSSet, InSARSet, JointModel, exp_cov, sample, hat_basis
from profil2d.insar import empirical_semivariogram, fit_exponential, combine_tracks


def header(path):
    h = open(path).readline()
    g = lambda k: float(re.search(rf"{k}=([-\d.]+)", h).group(1))
    faults = {m.group(1): float(m.group(2)) for m in re.finditer(r"fay\[([^\]]+)\]=([-\d.]+)", h)}
    return Profile(g("merkez_lon"), g("merkez_lat"), g("strike")), faults


def load_insar(path, xmax, ymax, cell):
    a = np.loadtxt(path, delimiter=",", comments="#")
    name = re.search(r"iz=(\w+)", open(path).readline()).group(1)
    x, y, los, pE, pN, pU = a[:, 2], a[:, 3], a[:, 4], a[:, 6], a[:, 7], a[:, 8]
    k = (np.abs(x) <= xmax) & (np.abs(y) <= ymax)
    x, y, los, pE, pN, pU = x[k], y[k], los[k], pE[k], pN[k], pU[k]
    ix, iy = np.floor(x / cell).astype(int), np.floor(y / cell).astype(int)
    keys = {}
    for i, key in enumerate(zip(ix, iy)): keys.setdefault(key, []).append(i)
    rows = [(x[v].mean(), y[v].mean(), np.median(los[v]), pE[v].mean(), pN[v].mean(), pU[v].mean(), len(v))
            for v in keys.values() if len(v) >= 2]
    r = np.array(rows)
    return name, r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], len(los)


def yatay_dataset(tracks, prof, cell, ymax):
    """(A) İki izin ortak hücrelerinde düşeyi tam eleyen kombinasyon (profil2d.insar.combine_tracks).
    Dönüş: (InSARSet 'yatay' [pE,pN = q, pU = 0], yarıvariogram parametreleri, κ = q·n)."""
    ta, td = tracks
    key = lambda t: {(int(np.floor(xx / cell)), int(np.floor(yy / cell))): i for i, (xx, yy) in enumerate(zip(t.x, t.y))}
    ka, kd = key(ta), key(td); com = sorted(set(ka) & set(kd))
    ia, id_ = np.array([ka[k] for k in com]), np.array([kd[k] for k in com])
    pa = np.array([ta.pE[ia], ta.pN[ia], ta.pU[ia]]); pd = np.array([td.pE[id_], td.pN[id_], td.pU[id_]])
    h_, q = combine_tracks(ta.los[ia], pa, td.los[id_], pd, prof.s)
    x, y = 0.5 * (ta.x[ia] + td.x[id_]), 0.5 * (ta.y[ia] + td.y[id_])
    far = np.abs(y) >= ymax / 2
    A = np.c_[np.ones(far.sum()), x[far], y[far]]; c, *_ = np.linalg.lstsq(A, h_[far], rcond=None)
    hh, gam, npair = empirical_semivariogram(x[far], y[far], h_[far] - A @ c, bins=np.arange(0, 42, 2.0), max_pairs=None)
    par = fit_exponential(hh, gam, npair)
    kap = q[0] * prof.n[0] + q[1] * prof.n[1]
    ds = InSARSet(name="yatay", x=x, y=y, los=h_, pE=q[0], pN=q[1], pU=np.zeros(len(h_)),
                  cov0=exp_cov(x, y, par["sigma"], par["lambda"], max(par["nugget"], 0.1)))
    return ds, par, kap


def summarize(samples, names, label):
    out = {}
    for j, nm in enumerate(names):
        v = samples[:, j]
        out[nm] = dict(median=float(np.median(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), mean=float(v.mean()), sd=float(v.std()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True); ap.add_argument("--insar", action="append", required=True)
    ap.add_argument("--model", choices=["tek", "iki"], default="tek"); ap.add_argument("--out", required=True)
    ap.add_argument("--xmax", type=float, default=50.0); ap.add_argument("--ymax", type=float, default=50.0)
    ap.add_argument("--cell", type=float, default=2.0); ap.add_argument("--knot", type=float, default=10.0)
    ap.add_argument("--nwalkers", type=int, default=32); ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yatay", action="store_true", help="A: iki izi düşeyden arındırılmış yatay bileşene birleştir")
    ap.add_argument("--Dmin", type=float, default=0.5); ap.add_argument("--Dmax", type=float, default=40.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof, fy = header(a.gps)

    g = np.genfromtxt(a.gps, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,   # 1: profil başlığı, 2: sütun adları
                      names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
    kg = (np.abs(g["x"]) <= a.xmax) & (np.abs(g["y"]) <= a.ymax)
    gps = GPSSet(y=g["y"][kg], vpar=g["vpar"][kg], vperp=g["vperp"][kg], spar=g["spar"][kg], sperp=g["sperp"][kg])
    print(f"Profil N{prof.strike:.1f}°D; kutu |x|≤{a.xmax:g}, |y|≤{a.ymax:g} km; GPS {kg.sum()} istasyon")

    tracks, covinfo = [], {}
    for path in a.insar:
        name, x, y, los, pE, pN, pU, n1 = load_insar(path, a.xmax, a.ymax, a.cell)
        far = np.abs(y) >= a.ymax / 2
        A = np.c_[np.ones(far.sum()), x[far], y[far]]; c, *_ = np.linalg.lstsq(A, los[far], rcond=None)
        h, gam, npair = empirical_semivariogram(x[far], y[far], los[far] - A @ c, bins=np.arange(0, 42, 2.0), max_pairs=None)
        par = fit_exponential(h, gam, npair)
        covinfo[name] = par
        C0 = exp_cov(x, y, par["sigma"], par["lambda"], max(par["nugget"], 0.1))
        tracks.append(InSARSet(name=name, x=x, y=y, los=los, pE=pE, pN=pN, pU=pU, cov0=C0))
        print(f"  InSAR {name}: 1 km blok {n1:,} -> {a.cell:g} km hücre {len(los):,};  C0: σ={par['sigma']:.2f} mm/yıl, λ={par['lambda']:.1f} km, nugget={par['nugget']:.2f}")

    if a.yatay:
        # A seçeneği: iki izi düşeyden arındırılmış faya paralel yatay bileşene birleştir
        assert len(tracks) == 2, "--yatay iki iz ister"
        ds, par, kap = yatay_dataset(tracks, prof, a.cell, a.ymax); tracks = [ds]; covinfo = {"yatay": par}
        print(f"  A: düşeyden arındırılmış yatay bileşen, ortak hücre {len(ds.los):,}; κ (faya dik sızma) medyan {np.median(kap):+.3f};"
              f" C0: σ={par['sigma']:.2f}, λ={par['lambda']:.1f} km, nugget={par['nugget']:.2f} mm/yıl")

    if a.model == "tek":
        faults = [dict(y0=float(np.mean(list(fy.values()))), free=True, y0_bounds=(-15.0, 15.0))]
    else:
        faults = [dict(y0=v, free=False, name=k) for k, v in fy.items()]
    knots = np.arange(-a.ymax, a.ymax + 0.1, a.knot)
    model = JointModel(prof.s, prof.n, gps, tracks, faults=faults, knots=knots, bounds=dict(D=(a.Dmin, a.Dmax)))
    print(f"D önseli: düzgün [{a.Dmin:g}, {a.Dmax:g}] km")
    print(f"Model '{a.model}': doğrusal olmayan {model.theta_names}; doğrusal {len(model.lin_names)} parametre (analitik integre)")

    t0 = time.time()
    out = sample(model, nwalkers=a.nwalkers, nsteps=a.nsteps, burn=a.burn, seed=a.seed, ndraw_lin=4000)
    print(f"Örnekleme {time.time() - t0:.0f} sn; ortalama kabul {np.mean(out['acc']):.2f}; otokorelasyon τ = "
          + ", ".join(f"{n}:{t:.0f}" for n, t in zip(model.theta_names, out["tau"])) + f"  (zincir uzunluğu {a.nsteps - a.burn})")

    th = out["theta"].copy(); tn = list(model.theta_names)
    for j, nm in enumerate(tn):
        if nm.startswith("log_"): th[:, j] = np.exp(th[:, j]); tn[j] = nm[4:]
    S_th = summarize(th, tn, "theta"); S_lin = summarize(out["lin"], model.lin_names, "lin")
    print("\nPosterior (medyan [%2.5–97.5]):")
    for nm in tn: print(f"  {nm:16s} {S_th[nm]['median']:8.2f}  [{S_th[nm]['lo']:7.2f} … {S_th[nm]['hi']:7.2f}]")
    for nm in [n for n in model.lin_names if n.startswith("V_")] + ["a_par", "a_perp", "b_perp"]:
        print(f"  {nm:16s} {S_lin[nm]['median']:8.3f}  [{S_lin[nm]['lo']:7.3f} … {S_lin[nm]['hi']:7.3f}]")
    for k in range(model.K):
        r = np.corrcoef(out["theta"][:, k], out["lin"][:, k])[0, 1]
        print(f"  korelasyon D_{k+1}–V_{k+1}: {r:+.2f}")
    if model.K == 2:
        vs = out["lin"][:, 0] + out["lin"][:, 1]
        print(f"  V_1+V_2 (toplam)   {np.median(vs):8.2f}  [{np.percentile(vs, 2.5):7.2f} … {np.percentile(vs, 97.5):7.2f}]")

    # rezidüeller (posterior medyan theta + koşullu ortalama m)
    th_med = np.median(out["theta"], axis=0); m_med, _ = model.conditional(th_med)
    pred = model.predict(th_med, m_med)
    D, y0, sf, s = model.unpack(th_med)
    res = dict(gps_par=float(np.sqrt(np.mean((gps.vpar - pred["gps_par"]) ** 2))), gps_perp=float(np.sqrt(np.mean((gps.vperp - pred["gps_perp"]) ** 2))))
    for t in tracks: res[t.name] = float(np.sqrt(np.mean((t.los - pred[t.name]) ** 2)))
    print("  rezidüel RMS (mm/yıl): " + ", ".join(f"{k} {v:.2f}" for k, v in res.items()))
    json.dump(dict(model=a.model, theta=S_th, lin=S_lin, cov=covinfo, rms=res, acc=float(np.mean(out["acc"])),
                   tau=[float(t) for t in out["tau"]], faults=faults, n_gps=int(kg.sum()), n_insar={t.name: len(t.los) for t in tracks}),
              open(f"{a.out}/ozet.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(f"{a.out}/posterior.npz", theta=out["theta"], lin=out["lin"], logp=out["logp"], chain=out["chain"],
                        theta_names=np.array(model.theta_names), lin_names=np.array(model.lin_names))

    # ---- figürler ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    yy = np.linspace(-a.ymax, a.ymax, 400)
    def vpar_curve(theta, m):
        D, y0, *_ = model.unpack(theta)
        return sum(m[k] * np.arctan((yy - y0[k]) / D[k]) / np.pi for k in range(model.K))
    pick = np.random.default_rng(1).choice(len(out["theta"]), min(300, len(out["theta"])), replace=False)
    curves = np.array([vpar_curve(out["theta"][i], out["lin"][i]) for i in pick])
    has_u = any(np.any(t.pU != 0) for t in tracks)
    fig, ax = plt.subplots(4 if has_u else 3, 1, figsize=(10, 13 if has_u else 10), sharex=True)
    ia = model.lin_names.index("a_par")
    ax[0].errorbar(gps.y, gps.vpar - m_med[ia], yerr=np.sqrt(gps.spar ** 2 + sf[0] ** 2), fmt="o", ms=4, color="#1f6feb", label="GPS faya paralel − a_par")
    ax[0].fill_between(yy, np.percentile(curves, 2.5, 0), np.percentile(curves, 97.5, 0), color="#1a7f37", alpha=.25, label="model %95")
    ax[0].plot(yy, np.median(curves, 0), color="#1a7f37", lw=2)
    ax[0].set_ylabel("mm/yıl"); ax[0].legend(fontsize=8)
    ib = model.lin_names.index("b_perp"); ip = model.lin_names.index("a_perp")
    ax[1].errorbar(gps.y, gps.vperp, yerr=np.sqrt(gps.sperp ** 2 + sf[1] ** 2), fmt="o", ms=4, color="#8250df", label="GPS faya dik")
    ax[1].plot(yy, m_med[ip] + m_med[ib] * yy, "k-", label="model (bölgesel doğrusal)"); ax[1].set_ylabel("mm/yıl"); ax[1].legend(fontsize=8)
    ax[-1].set_xlabel("y (km, + kuzey)")
    bins = np.arange(-a.ymax, a.ymax + 0.1, 4.0); mid = 0.5 * (bins[1:] + bins[:-1])
    for t, tr in enumerate(tracks):
        j = model.K + 3 + 3 * t
        ramp = m_med[j] + m_med[j + 1] * tr.x + m_med[j + 2] * tr.y
        perp = (tr.pE * prof.n[0] + tr.pN * prof.n[1]) * m_med[ib] * tr.y
        dd = tr.los - ramp - perp
        med = [np.median(dd[(tr.y >= b0) & (tr.y < b1)]) if np.any((tr.y >= b0) & (tr.y < b1)) else np.nan for b0, b1 in zip(bins[:-1], bins[1:])]
        sp = np.mean(tr.pE * prof.s[0] + tr.pN * prof.s[1]); pu = np.mean(tr.pU)
        u = hat_basis(yy, knots) @ m_med[model.iu]
        ax[2].plot(mid, med, "o-", ms=3, label=f"{tr.name}: veri − rampa − dik terim (4 km medyan)")
        ax[2].plot(yy, sp * np.median(curves, 0) + pu * u, "--", lw=1.5, label=f"{tr.name}: model (fay + düşey)")
    ax[2].set_ylabel("LOS mm/yıl"); ax[2].legend(fontsize=7)
    if has_u:
        u_draws = np.array([hat_basis(yy, knots) @ out["lin"][i, model.iu] for i in pick])
        ax[3].fill_between(yy, np.percentile(u_draws, 2.5, 0), np.percentile(u_draws, 97.5, 0), color="#bf8700", alpha=.3)
        ax[3].plot(yy, np.median(u_draws, 0), color="#bf8700", lw=2, label="düşey nuisance u(y) (iki izin ortağı)")
        ax[3].set_ylabel("mm/yıl"); ax[3].set_xlabel("y (km, + kuzey)"); ax[3].legend(fontsize=8)
    for f in fy.values():
        for x_ in ax: x_.axvline(f, ls="--", color="0.5", lw=.8)
    for x_ in ax: x_.grid(alpha=.3)
    fig.suptitle(f"2B ortak çözüm — model '{a.model}'"); fig.tight_layout(); fig.savefig(f"{a.out}/uyum.png", dpi=110)

    fig, ax = plt.subplots(1, model.K + 1, figsize=(5 * (model.K + 1), 4.2))
    ax = np.atleast_1d(ax)
    for k in range(model.K):
        ax[k].scatter(out["theta"][:, k], out["lin"][:, k], s=2, alpha=.3)
        ax[k].set_xlabel(f"D_{k+1} (km)"); ax[k].set_ylabel(f"V_{k+1} (mm/yıl, + sağ-yanal)"); ax[k].grid(alpha=.3)
    for j, nm in enumerate(tn):
        if nm.startswith(("floor", "s_")):
            ax[-1].hist(th[:, j], bins=40, histtype="step", label=nm)
    ax[-1].set_xlabel("gürültü parametreleri"); ax[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.out}/posterior.png", dpi=110)
    print(f"\nçıktılar: {a.out}/ozet.json, posterior.npz, uyum.png, posterior.png")


if __name__ == "__main__":
    main()
