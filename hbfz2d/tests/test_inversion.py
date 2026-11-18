"""2B ortak ters çözüm (GPS + InSAR izleri): marjinal olabilirlik ve sentetik geri alma."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from profil2d.inversion import GPSSet, InSARSet, JointModel, exp_cov, sample
from profil2d.forward import los_unit_vector

S, N = np.array([1.0, 0.0]), np.array([0.0, 1.0])       # strike = 90°: s doğu, n kuzey
P_ASC, P_DSC = los_unit_vector(41.0, -13.4), los_unit_vector(37.0, -166.6)


def _hat(y, knots):
    B = np.zeros((len(y), len(knots)))
    for m, k in enumerate(knots):
        h = knots[1] - knots[0]
        B[:, m] = np.clip(1 - np.abs(y - k) / h, 0, None)
    return B


def synthetic(V=3.0, D=12.0, y0=1.0, ng=50, ni=400, seed=0, sig=1.5, lam=8.0, nug=0.3, gps_sd=0.5):
    rng = np.random.default_rng(seed)
    knots = np.arange(-50, 51, 10.0)
    yg = np.sort(rng.uniform(-50, 50, ng))
    g = lambda y: V / np.pi * np.arctan((y - y0) / D)
    gps = GPSSet(y=yg, vpar=g(yg) + 4.0 + rng.normal(0, gps_sd, ng), vperp=-1.0 + 0.03 * yg + rng.normal(0, gps_sd, ng),
                 spar=np.full(ng, 0.05), sperp=np.full(ng, 0.05))
    u_true = rng.normal(0, 1.5, len(knots))
    tracks = []
    for name, p, ramp in (("asc", P_ASC, (1.0, 0.01, -0.02)), ("dsc", P_DSC, (-0.5, -0.015, 0.01))):
        x = rng.uniform(-50, 50, ni); y = rng.uniform(-50, 50, ni)
        C0 = exp_cov(x, y, sig, lam, nug)
        sp, np_ = p[0] * S[0] + p[1] * S[1], p[0] * N[0] + p[1] * N[1]
        los = sp * g(y) + np_ * 0.03 * y + p[2] * (_hat(y, knots) @ u_true) + ramp[0] + ramp[1] * x + ramp[2] * y
        los = los + np.linalg.cholesky(C0) @ rng.normal(size=ni)
        tracks.append(InSARSet(name=name, x=x, y=y, los=los, pE=np.full(ni, p[0]), pN=np.full(ni, p[1]), pU=np.full(ni, p[2]), cov0=C0))
    return gps, tracks, knots


def test_marginal_matches_dense_formula():
    gps, tracks, knots = synthetic(ng=6, ni=12, seed=4)
    m = JointModel(S, N, gps, tracks, faults=[dict(y0=0.5, free=True), dict(y0=-3.0, free=False)], knots=knots)
    theta = np.array([11.0, 9.0, 1.5, np.log(0.7), np.log(1.6), np.log(1.3), np.log(0.8)])   # D1, D2, y0_1, log σ_par, log σ_perp, log s_asc, log s_dsc
    assert m.theta_names == ["D_1", "D_2", "y0_1", "log_floor_par", "log_floor_perp", "log_s_asc", "log_s_dsc"]
    G, d, C, Psd = m.dense(theta)
    ref = multivariate_normal(mean=np.zeros(len(d)), cov=C + G @ np.diag(Psd ** 2) @ G.T).logpdf(d)
    assert m.log_marginal(theta) == pytest.approx(ref, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("V", [3.0, -2.0])
def test_recovers_synthetic_single_fault(V):
    gps, tracks, knots = synthetic(V=V, D=12.0, y0=1.0, seed=1)
    m = JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=False)], knots=knots)
    hyp = [np.log(0.5), np.log(0.5), 0.0, 0.0]           # doğru gürültü değerleri
    Ds = np.arange(4.0, 40.1, 1.0)
    lm = np.array([m.log_marginal(np.r_[D, hyp]) for D in Ds])
    # Bu gürültüde D zayıf çözülür (V–D ödünleşimi); yalnızca gerçek D'nin DIŞLANMADIĞINI isteriz
    assert lm[Ds == 12.0][0] >= lm.max() - 2.0
    mean, cov = m.conditional(np.r_[12.0, hyp])
    iV = m.lin_names.index("V_1")
    assert abs(mean[iV] - V) < 3 * np.sqrt(cov[iV, iV]) and np.sqrt(cov[iV, iV]) < 1.0
    assert np.sign(mean[iV]) == np.sign(V)


def test_sampler_smoke():
    gps, tracks, knots = synthetic(ng=20, ni=80, seed=2)
    m = JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=True)], knots=knots)
    out = sample(m, nwalkers=12, nsteps=40, burn=10, seed=0, ndraw_lin=50)
    assert out["theta"].shape[1] == len(m.theta_names)
    assert out["lin"].shape == (out["theta"].shape[0], len(m.lin_names))
    assert np.all(np.isfinite(out["logp"]))


def test_recovers_depth_with_low_noise():
    """Gürültü düşükken D gerçekten geri alınmalı (kod D'ye duyarlı)."""
    gps, tracks, knots = synthetic(V=3.0, D=12.0, y0=1.0, seed=5, sig=0.3, nug=0.1, gps_sd=0.15)
    m = JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=False)], knots=knots)
    hyp = [np.log(0.15), np.log(0.15), 0.0, 0.0]
    Ds = np.arange(4.0, 40.1, 0.5)
    lm = np.array([m.log_marginal(np.r_[D, hyp]) for D in Ds])
    assert 9.0 <= Ds[np.argmax(lm)] <= 15.0
    assert lm.max() - lm[Ds == 30.0][0] > 4.0                # derin çözüm açıkça daha kötü
