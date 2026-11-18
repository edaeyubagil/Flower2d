#!/usr/bin/env python3
"""Veri setlerinin katkısı: aynı tek-fay modelini yalnızca GPS ya da yalnızca InSAR ile çöz.
Kullanım: python scripts/katki_testi.py --gps local/gps_profil.csv --insar A.csv --insar B.csv --veri gps|insar --out DIR
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from ters_coz import header, load_insar, yatay_dataset
from profil2d.inversion import GPSSet, InSARSet, JointModel, exp_cov, sample
from profil2d.insar import empirical_semivariogram, fit_exponential

ap = argparse.ArgumentParser()
ap.add_argument("--gps", required=True); ap.add_argument("--insar", action="append", default=[])
ap.add_argument("--veri", choices=["gps", "insar", "yatay"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nsteps", type=int, default=4000); ap.add_argument("--burn", type=int, default=1500)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prof, fy = header(a.gps)
g = np.genfromtxt(a.gps, delimiter=",", comments="#", dtype=None, encoding="utf-8", skip_header=2,
                  names=["site", "lon", "lat", "x", "y", "ve", "vn", "se", "sn", "rho", "vpar", "vperp", "spar", "sperp", "rpp"])
kg = (np.abs(g["x"]) <= 50) & (np.abs(g["y"]) <= 50)
if a.veri == "gps":
    gps = GPSSet(y=g["y"][kg], vpar=g["vpar"][kg], vperp=g["vperp"][kg], spar=g["spar"][kg], sperp=g["sperp"][kg]); tracks = []
else:   # insar ya da yatay: GPS yok
    gps = GPSSet(y=np.zeros(0), vpar=np.zeros(0), vperp=np.zeros(0), spar=np.zeros(0), sperp=np.zeros(0)); tracks = []
    for p in a.insar:
        name, x, y, los, pE, pN, pU, _ = load_insar(p, 50, 50, 2.0)
        far = np.abs(y) >= 25; A = np.c_[np.ones(far.sum()), x[far], y[far]]; c, *_ = np.linalg.lstsq(A, los[far], rcond=None)
        h, gam, npair = empirical_semivariogram(x[far], y[far], los[far] - A @ c, bins=np.arange(0, 42, 2.0), max_pairs=None)
        par = fit_exponential(h, gam, npair)
        tracks.append(InSARSet(name, x, y, los, pE, pN, pU, exp_cov(x, y, par["sigma"], par["lambda"], max(par["nugget"], 0.1))))
    if a.veri == "yatay":
        tracks = [yatay_dataset(tracks, prof, 2.0, 50.0)[0]]
model = JointModel(prof.s, prof.n, gps, tracks, faults=[dict(y0=float(np.mean(list(fy.values()))), free=True, y0_bounds=(-15.0, 15.0))],
                   knots=np.arange(-50, 50.1, 10.0))
out = sample(model, nwalkers=32, nsteps=a.nsteps, burn=a.burn, seed=0, ndraw_lin=4000)
D, V = out["theta"][:, 0], out["lin"][:, 0]
res = dict(veri=a.veri, V=[float(np.median(V)), float(np.percentile(V, 2.5)), float(np.percentile(V, 97.5))],
           D=[float(np.median(D)), float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))],
           y0=[float(np.median(out["theta"][:, 1])), float(np.percentile(out["theta"][:, 1], 2.5)), float(np.percentile(out["theta"][:, 1], 97.5))],
           P_Vpos=float(np.mean(V > 0)), tau=[float(t) for t in out["tau"]], acc=float(np.mean(out["acc"])),
           corr_DV=float(np.corrcoef(D, V)[0, 1]), D_hist=np.histogram(D, bins=np.arange(0.5, 40.6, 5))[0].tolist())
json.dump(res, open(f"{a.out}/ozet.json", "w"), indent=1)
np.savez_compressed(f"{a.out}/posterior.npz", theta=out["theta"], lin=out["lin"], theta_names=np.array(model.theta_names), lin_names=np.array(model.lin_names))
print(json.dumps(res, ensure_ascii=False))
