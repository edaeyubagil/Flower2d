"""2B ortak ters çözüm: GPS (faya paralel + dik) ve InSAR izleri (LOS), ortak fiziksel model.

Her veri seti KENDİ gözlem uzayında öngörülür (GPS ile LOS birbirine çevrilmez):

  GPS paralel : sum_k V_k g_k(y) + a_par
  GPS dik     : a_perp + b_perp * y
  InSAR (iz t): (s·p) sum_k V_k g_k(y) + (n·p) b_perp y + pU * u(y) + c0_t + cx_t x + cy_t y
  g_k(y) = (1/pi) atan((y - y0_k) / D_k);  V > 0 SAĞ-YANAL  (profil2d.forward ile aynı uzlaşım)
  u(y)   = sum_m u_m B_m(y), B_m: düğümlerde şapka (doğrusal) fonksiyonları; iki izin ortak düşey nuisance'ı.

Doğrusal parametreler m (V, ofsetler, gradyan, rampalar, u) sıfır ortalamalı Gauss önseliyle ANALİTİK
olarak integre edilir; doğrusal olmayanlar theta = [D_k..., serbest y0..., log σ_taban_par, log σ_taban_perp, log s_t...]
örneklenir. Gürültü: GPS σ² = formal² + taban² (paralel ve dik için AYRI taban);  InSAR C_t = s_t² · C0_t (C0: yarıvariogramdan).

log p(d|theta) = -1/2 [ dᵀC⁻¹d − bᵀA⁻¹b + log|C| + log|P| + log|A| + N log 2π ],
A = GᵀC⁻¹G + P⁻¹,  b = GᵀC⁻¹d.  m | theta, d ~ N(A⁻¹b, A⁻¹).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


@dataclass
class GPSSet:
    y: np.ndarray
    vpar: np.ndarray
    vperp: np.ndarray
    spar: np.ndarray
    sperp: np.ndarray
    x: np.ndarray = None          # yalnız 3B model (model3d) için gerekli


@dataclass
class InSARSet:
    name: str
    x: np.ndarray
    y: np.ndarray
    los: np.ndarray
    pE: np.ndarray
    pN: np.ndarray
    pU: np.ndarray
    cov0: np.ndarray


def exp_cov(x, y, sigma, lam, nugget=0.0):
    """C(h) = σ² exp(−h/λ) + nugget² δ."""
    h = np.hypot(np.subtract.outer(x, x), np.subtract.outer(y, y))
    return sigma ** 2 * np.exp(-h / lam) + nugget ** 2 * np.eye(len(x))


def hat_basis(y, knots):
    if len(knots) == 0:
        return np.zeros((len(y), 0))
    h = np.diff(knots)[0]
    return np.clip(1.0 - np.abs(np.subtract.outer(y, knots)) / h, 0.0, None)


DEFAULT_PRIOR_SD = dict(V=10.0, a_par=50.0, a_perp=50.0, b_perp=0.2, c0=20.0, cx=0.5, cy=0.5, u=10.0)
DEFAULT_BOUNDS = dict(D=(0.5, 40.0), y0=(-10.0, 10.0), log_floor=(np.log(0.05), np.log(5.0)), log_s=(np.log(0.1), np.log(10.0)))


class JointModel:
    def __init__(self, s, n, gps, tracks, faults, knots, prior_sd=None, bounds=None):
        self.s, self.n = np.asarray(s, float), np.asarray(n, float)
        self.gps, self.tracks, self.faults = gps, list(tracks), [dict(f) for f in faults]
        self.knots = np.asarray(knots, float)
        self.psd = dict(DEFAULT_PRIOR_SD, **(prior_sd or {}))
        self.bounds = dict(DEFAULT_BOUNDS, **(bounds or {}))
        K, T = len(self.faults), len(self.tracks)
        self.theta_names = ([f"D_{k+1}" for k in range(K)] + [f"y0_{k+1}" for k, f in enumerate(self.faults) if f.get("free")]
                            + ["log_floor_par", "log_floor_perp"] + [f"log_s_{t.name}" for t in self.tracks])
        self.lin_names = ([f"V_{k+1}" for k in range(K)] + ["a_par", "a_perp", "b_perp"]
                          + [f"{c}_{t.name}" for t in self.tracks for c in ("c0", "cx", "cy")]
                          + [f"u_{m}" for m in range(len(self.knots))])
        self.K, self.T = K, T
        self.nlin = len(self.lin_names)
        sd = [self.psd["V"]] * K + [self.psd["a_par"], self.psd["a_perp"], self.psd["b_perp"]]
        sd += [self.psd[c] for _ in self.tracks for c in ("c0", "cx", "cy")] + [self.psd["u"]] * len(self.knots)
        self.prior_sd = np.array(sd, float)
        self._prep()

    # ---- sabit (theta'dan bağımsız) kısımlar ----
    def _fixed_cols(self, which, t=None):
        """V sütunları hariç tasarım matrisi (V sütunları sıfır bırakılır)."""
        K = self.K
        if which == "gps":
            g = self.gps; ng = len(g.y)
            G = np.zeros((2 * ng, self.nlin))
            G[:ng, K + 0] = 1.0
            G[ng:, K + 1] = 1.0; G[ng:, K + 2] = g.y
            return G
        tr = self.tracks[t]; ni = len(tr.y)
        G = np.zeros((ni, self.nlin))
        np_ = tr.pE * self.n[0] + tr.pN * self.n[1]
        G[:, K + 2] = np_ * tr.y
        j = K + 3 + 3 * t
        G[:, j], G[:, j + 1], G[:, j + 2] = 1.0, tr.x, tr.y
        G[:, K + 3 + 3 * self.T:] = tr.pU[:, None] * hat_basis(tr.y, self.knots)
        return G

    def _prep(self):
        g = self.gps
        self.d_gps = np.r_[g.vpar, g.vperp]
        self.s2_gps = np.r_[g.spar, g.sperp] ** 2
        self.G_gps = self._fixed_cols("gps")
        self.tr_pre = []
        for t, tr in enumerate(self.tracks):
            L = np.linalg.cholesky(tr.cov0)
            WG = solve_triangular(L, self._fixed_cols("insar", t), lower=True)
            Wd = solve_triangular(L, tr.los, lower=True)
            sp = tr.pE * self.s[0] + tr.pN * self.s[1]
            self.tr_pre.append(dict(L=L, WG=WG, Wd=Wd, sp=sp, logdet0=2 * np.sum(np.log(np.diag(L))), n=len(tr.y)))

    # ---- theta ----
    def unpack(self, theta):
        th = np.asarray(theta, float); K = self.K
        D = th[:K]; i = K; y0 = []
        for f in self.faults:
            if f.get("free"):
                y0.append(th[i]); i += 1
            else:
                y0.append(f["y0"])
        return D, np.array(y0), np.exp(th[i:i + 2]), np.exp(th[i + 2:i + 2 + self.T])

    def log_prior(self, theta):
        D, y0, sf, s = self.unpack(theta); b = self.bounds
        th = np.asarray(theta, float); K = self.K
        if np.any(D < b["D"][0]) or np.any(D > b["D"][1]): return -np.inf
        for k, f in enumerate(self.faults):
            if f.get("free"):
                lo, hi = f.get("y0_bounds", b["y0"])
                if not (lo <= y0[k] <= hi): return -np.inf
        lf = np.log(sf)
        if np.any(lf < b["log_floor"][0]) or np.any(lf > b["log_floor"][1]): return -np.inf
        if np.any(np.log(s) < b["log_s"][0]) or np.any(np.log(s) > b["log_s"][1]): return -np.inf
        return 0.0

    def _floor2(self, sf):
        ng = len(self.gps.y)
        return np.r_[np.full(ng, sf[0] ** 2), np.full(ng, sf[1] ** 2)]

    def _vcols(self, y, D, y0):
        return np.stack([np.arctan((y - y0[k]) / D[k]) / np.pi for k in range(self.K)], axis=1).reshape(len(y), self.K)

    # ---- fay (V) sütunları: alt sınıflar (model3d.JointModel3D) bunları değiştirir ----
    def _fault_columns(self, theta):
        """V = 1 için fay katkısı: (GPS [paralel; dik] (2ng, K), [iz t için (n_t, K)])."""
        D, y0, *_ = self.unpack(theta)
        ng = len(self.gps.y)
        Vg = np.r_[self._vcols(self.gps.y, D, y0), np.zeros((ng, self.K))]
        Vt = [pre["sp"][:, None] * self._vcols(tr.y, D, y0) for tr, pre in zip(self.tracks, self.tr_pre)]
        return Vg, Vt

    def _fault_columns_white(self, theta):
        """Aynı sütunlar; InSAR kısmı C0'ın Cholesky çarpanıyla beyazlatılmış (L⁻¹ V)."""
        Vg, Vt = self._fault_columns(theta)
        return Vg, [solve_triangular(pre["L"], v, lower=True) for v, pre in zip(Vt, self.tr_pre)]

    def _normal_eq(self, theta):
        D, y0, sf, s = self.unpack(theta)
        Vg, WVt = self._fault_columns_white(theta)
        A = np.diag(1.0 / self.prior_sd ** 2); b = np.zeros(self.nlin); quad = 0.0; logdet = 0.0; ntot = 0
        # GPS (köşegen)
        G = self.G_gps.copy()
        G[:, :self.K] = Vg
        w = 1.0 / (self.s2_gps + self._floor2(sf))
        A += G.T @ (G * w[:, None]); b += G.T @ (w * self.d_gps); quad += np.sum(w * self.d_gps ** 2)
        logdet += np.sum(np.log(1.0 / w)); ntot += len(w)
        # InSAR (beyazlatılmış, ölçekli)
        for t, pre in enumerate(self.tr_pre):
            WG = pre["WG"].copy()
            WG[:, :self.K] = WVt[t]
            f = 1.0 / s[t] ** 2
            A += f * (WG.T @ WG); b += f * (WG.T @ pre["Wd"]); quad += f * (pre["Wd"] @ pre["Wd"])
            logdet += pre["logdet0"] + 2 * pre["n"] * np.log(s[t]); ntot += pre["n"]
        return A, b, quad, logdet, ntot

    def log_marginal(self, theta):
        A, b, quad, logdet, ntot = self._normal_eq(theta)
        cf = cho_factor(A, lower=True)
        logdetA = 2 * np.sum(np.log(np.diag(cf[0])))
        return -0.5 * (quad - b @ cho_solve(cf, b) + logdet + np.sum(np.log(self.prior_sd ** 2)) + logdetA + ntot * np.log(2 * np.pi))

    def log_post(self, theta):
        lp = self.log_prior(theta)
        return lp + self.log_marginal(theta) if np.isfinite(lp) else -np.inf

    def conditional(self, theta):
        A, b, *_ = self._normal_eq(theta)
        cf = cho_factor(A, lower=True)
        cov = cho_solve(cf, np.eye(self.nlin))
        return cov @ b, cov

    def predict(self, theta, m):
        """Ortalama tahminler: dict(gps_par, gps_perp, <iz>) aynı veri sırasıyla."""
        Vg, Vt = self._fault_columns(theta)
        G = self.G_gps.copy(); ng = len(self.gps.y)
        G[:, :self.K] = Vg
        pg = G @ m; out = dict(gps_par=pg[:ng], gps_perp=pg[ng:])
        for t, tr in enumerate(self.tracks):
            Gt = self._fixed_cols("insar", t)
            Gt[:, :self.K] = Vt[t]
            out[tr.name] = Gt @ m
        return out

    def dense(self, theta):
        """Test için: tam G, d, C ve önsel sd (yoğun formülle karşılaştırma)."""
        D, y0, sf, s = self.unpack(theta)
        Vg, Vt = self._fault_columns(theta)
        Gs, ds, Cs = [], [], []
        G = self.G_gps.copy(); G[:, :self.K] = Vg
        Gs.append(G); ds.append(self.d_gps); Cs.append(np.diag(self.s2_gps + self._floor2(sf)))
        for t, tr in enumerate(self.tracks):
            Gt = self._fixed_cols("insar", t); Gt[:, :self.K] = Vt[t]
            Gs.append(Gt); ds.append(tr.los); Cs.append(s[t] ** 2 * tr.cov0)
        from scipy.linalg import block_diag
        return np.vstack(Gs), np.concatenate(ds), block_diag(*Cs), self.prior_sd


def sample(model, nwalkers=32, nsteps=3000, burn=1000, thin=1, seed=0, ndraw_lin=2000, start=None, progress=False):
    """emcee ile theta örnekle; her theta örneği için doğrusal parametreleri koşullu Gauss'tan çek.
    Dönüş: dict(theta, lin, logp, acc, tau, chain)."""
    import emcee
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    nd = len(model.theta_names)
    if start is None:
        start = []
        for nm in model.theta_names:
            start.append(float(np.mean(model.bounds["D"])) if (nm == "D" or nm.startswith("D_")) else 0.0 if nm.startswith("y0") else np.log(0.5) if nm.startswith("log_floor") else 0.0)
        start = np.array(start)
        r = minimize(lambda th: -model.log_post(th) if np.isfinite(model.log_post(th)) else 1e30, start, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-3, fatol=1e-3))
        start = r.x
    p0 = start + 1e-2 * rng.normal(size=(nwalkers, nd))
    for i in range(nwalkers):
        while not np.isfinite(model.log_post(p0[i])):
            p0[i] = start + 1e-3 * rng.normal(size=nd)
    sampler = emcee.EnsembleSampler(nwalkers, nd, model.log_post)
    sampler.run_mcmc(p0, nsteps, progress=progress)
    chain = sampler.get_chain(discard=burn, thin=thin)                 # (adım, walker, nd)
    flat = chain.reshape(-1, nd); logp = sampler.get_log_prob(discard=burn, thin=thin).reshape(-1)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(nd, np.nan)
    idx = rng.choice(len(flat), size=min(ndraw_lin, len(flat)), replace=False)
    th = flat[idx]; lin = np.empty((len(idx), model.nlin))
    for j, t in enumerate(th):
        mu, cov = model.conditional(t)
        lin[j] = rng.multivariate_normal(mu, cov, method="cholesky")
    return dict(theta=th, lin=lin, logp=logp[idx], acc=sampler.acceptance_fraction, tau=tau, chain=chain, start=start)
