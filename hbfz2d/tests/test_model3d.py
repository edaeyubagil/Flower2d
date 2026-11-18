"""3B segment modeli (C): birleşik fay izi, Okada Green fonksiyonları, kurgular ve ortak ters çözüm.

Kurgular (V = 1, sağ-yanal; profil çerçevesi x = s, y = n):
  derin     : yalnız haritalı iz boyunca D'nin altında kayma
  blok      : iz + uçlardan uzatma boyunca D'nin altında kayma
              = [iz+uzatma, 0..∞ blok kayması] − [iz+uzatma, 0..D geri kayma]
  blok_krip : [iz+uzatma, 0..∞] − [yalnız iz, 0..D]  (uzatmalar yüzeyde kripli)
"""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from profil2d.forward import screw
from profil2d.inversion import GPSSet, InSARSet, JointModel, exp_cov, sample
from profil2d.okada import okada_rect
from profil2d.model3d import (merged_trace, trace_pieces, fault_green, make_green, DepthSpline, JointModel3D, BIG)

S, N = np.array([1.0, 0.0]), np.array([0.0, 1.0])       # profil doğrultusu 90°: x doğu, y kuzey


# ------------------------------------------------------------------------------------------
# birleşik iz
# ------------------------------------------------------------------------------------------
def test_merged_trace_straight_line_is_exact():
    x = np.linspace(-20, 30, 7); y = 3 + 0.1 * x
    xn, yn = merged_trace([np.c_[x, y]], step=5.0)
    assert xn[0] == pytest.approx(-20) and xn[-1] == pytest.approx(30)
    assert np.all(np.diff(xn) <= 5.0 + 1e-9) and np.all(np.diff(xn) > 0)
    assert np.allclose(yn, 3 + 0.1 * xn, atol=1e-9)


def test_merged_trace_merges_parallel_strands():
    """Bindirmeli paralel iki kol (y = ±2) tek bir izde birleşir (ortalama); tek kolun olduğu yerde o kol."""
    a = np.c_[np.linspace(-10, 10, 5), np.full(5, 2.0)]
    b = np.c_[np.linspace(-10, 20, 7), np.full(7, -2.0)]
    xn, yn = merged_trace([a, b], step=2.0)
    both = (xn >= -8) & (xn <= 8); only_b = xn >= 13
    assert np.allclose(yn[both], 0.0, atol=1e-9)
    assert np.allclose(yn[only_b], -2.0, atol=1e-9)


def test_merged_trace_bridges_gap():
    a = np.c_[np.linspace(-20, -6, 5), np.full(5, 1.0)]
    b = np.c_[np.linspace(6, 20, 5), np.full(5, 1.0)]
    xn, yn = merged_trace([a, b], step=2.0)
    assert np.all(np.isfinite(yn)) and np.allclose(yn, 1.0)
    assert xn[0] == pytest.approx(-20) and xn[-1] == pytest.approx(20)


def test_trace_pieces_contiguous_with_extensions():
    xn, yn = np.array([-10.0, 0.0, 10.0]), np.array([0.0, 0.0, 2.0])
    tr, ex = trace_pieces(xn, yn, ext=100.0)
    assert len(tr) == 2 and len(ex) == 2
    ends = [(p["x0"], p["y0"], p["x1"], p["y1"]) for p in tr]
    assert ends[0][2:] == ends[1][:2]                                  # bitişik
    assert sum(p["length"] for p in tr) == pytest.approx(10 + np.hypot(10, 2))
    assert ex[0]["length"] == pytest.approx(100) and ex[1]["length"] == pytest.approx(100)
    assert (ex[0]["x1"], ex[0]["y1"]) == pytest.approx((-10.0, 0.0))  # ilk uzatma izin başına bağlanır
    assert (ex[1]["x0"], ex[1]["y0"]) == pytest.approx((10.0, 2.0))   # son uzatma izin sonundan başlar
    # varsayılan: zon doğrultusu (profil +x); izin yönü ters sayısallaştırılsa da uzatma dışa doğru
    assert np.allclose([ex[0]["x1"] - ex[0]["x0"], ex[0]["y1"] - ex[0]["y0"]], [100.0, 0.0])
    assert np.allclose([ex[1]["x1"] - ex[1]["x0"], ex[1]["y1"] - ex[1]["y0"]], [100.0, 0.0])
    _, exr = trace_pieces(xn[::-1], yn[::-1], ext=100.0)
    assert np.allclose([exr[1]["x1"], exr[1]["y1"]], [-110.0, 0.0])
    chord = np.array([20.0, 2.0]) / np.hypot(20, 2)
    _, exk = trace_pieces(xn, yn, ext=100.0, direction="kiris")
    assert np.allclose([exk[1]["x1"] - exk[1]["x0"], exk[1]["y1"] - exk[1]["y0"]], 100 * chord)
    assert tr[0]["strike"] == pytest.approx(90.0)                       # +x yönü = 90°
    assert trace_pieces(xn, yn, ext=0.0)[1] == []


# ------------------------------------------------------------------------------------------
# Green fonksiyonları ve kurgular
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("D", [10.0, 15.0])
def test_blok_long_straight_trace_is_2d_screw(D):
    """Düz uzun iz (y = 2), sonsuz uzatma: profilde faya paralel = (1/pi) atan((y-2)/D), dik ~0."""
    tr, ex = trace_pieces(np.array([-30.0, 0.0, 30.0]), np.array([2.0, 2.0, 2.0]), ext=1e6)
    y = np.linspace(-50, 50, 21); x = np.zeros_like(y)
    ux, uy, uz = fault_green(x, y, tr, ex, D, "blok")
    assert np.allclose(ux, screw(y, 2.0, D, 1.0), atol=3e-4)
    assert np.allclose(uy, 0.0, atol=1e-6) and np.allclose(uz, 0.0, atol=1e-6)


def _literal(x, y, pieces_long, pieces_back, D):
    """[pieces_long, 0..BIG] − [pieces_back, 0..D] (doğrudan Okada toplamı)."""
    out = np.zeros((3, len(x)))
    for p in pieces_long:
        out += np.array(okada_rect(x, y, p["xm"], p["ym"], p["strike"], p["length"], 0.0, BIG, ss=1.0))
    for p in pieces_back:
        out -= np.array(okada_rect(x, y, p["xm"], p["ym"], p["strike"], p["length"], 0.0, D, ss=1.0))
    return out


def test_formulations_equal_longterm_minus_backslip():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-60, 60, 200), rng.uniform(-40, 40, 200)
    tr, ex = trace_pieces(np.array([-25.0, -5.0, 10.0, 22.0]), np.array([3.0, 4.0, 1.0, -3.0]), ext=100.0)
    D = 12.0
    blok = np.array(fault_green(x, y, tr, ex, D, "blok"))
    krip = np.array(fault_green(x, y, tr, ex, D, "blok_krip"))
    assert np.allclose(blok, _literal(x, y, tr + ex, tr + ex, D), atol=1e-8)
    assert np.allclose(krip, _literal(x, y, tr + ex, tr, D), atol=1e-8)


def test_derin_is_tip_limited():
    """30 km'lik izde derin kurgu: merkezde 2B'den küçük; uçların çok ötesinde ~0. blok (100 km) ise uçların
    ötesinde de uzatma boyunca bir hız farkı taşır."""
    tr, ex = trace_pieces(np.array([-15.0, 0.0, 15.0]), np.zeros(3), ext=100.0)
    D = 12.0
    yc = np.array([-D, D]); xc = np.zeros(2)
    der = fault_green(xc, yc, tr, ex, D, "derin")[0]
    assert 0 < der[1] < 0.6 * screw(D, 0.0, D, 1.0) and der[0] == pytest.approx(-der[1], abs=1e-9)
    xf = np.array([80.0, 80.0]); yf = np.array([-20.0, 20.0])
    der_far = fault_green(xf, yf, tr, ex, D, "derin")[0]
    blk_far = fault_green(xf, yf, tr, ex, D, "blok")[0]
    assert np.abs(der_far[1] - der_far[0]) < 0.05
    assert blk_far[1] - blk_far[0] > 0.3 and blk_far[1] - blk_far[0] > 5 * abs(der_far[1] - der_far[0])


def test_blok_krip_has_surface_step_on_extension():
    """Uzatma boyunca (iz ucunun ötesinde) yüzeyde krip: blok_krip'te ~1 (V) basamak, blok'ta yok."""
    tr, ex = trace_pieces(np.array([-15.0, 0.0, 15.0]), np.zeros(3), ext=100.0)
    D = 12.0; eps = 0.05
    x = np.array([40.0, 40.0]); y = np.array([-eps, eps])
    jump_k = np.diff(fault_green(x, y, tr, ex, D, "blok_krip")[0])[0]
    jump_b = np.diff(fault_green(x, y, tr, ex, D, "blok")[0])[0]
    assert jump_k == pytest.approx(1.0, abs=0.02)
    assert abs(jump_b) < 0.01


def test_make_green_columns_per_fault():
    f1 = trace_pieces(np.array([-20.0, 20.0]), np.array([3.0, 3.0]), ext=50.0)
    f2 = trace_pieces(np.array([0.0, 40.0]), np.array([-2.0, -2.0]), ext=50.0)
    x, y = np.array([0.0, 10.0, 30.0]), np.array([1.0, -5.0, 0.0])
    g = make_green(x, y, [f1, f2], "blok")
    Ux, Uy, Uz = g(12.0)
    assert Ux.shape == (3, 2)
    assert np.allclose(Ux[:, 1], fault_green(x, y, *f2, 12.0, "blok")[0])


def test_depth_spline_accuracy():
    tr, ex = trace_pieces(np.array([-25.0, -5.0, 10.0, 22.0]), np.array([3.0, 4.0, 1.0, -3.0]), ext=100.0)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-50, 50, 300), rng.uniform(-50, 50, 300)
    f = lambda D: np.array(fault_green(x, y, tr, ex, D, "blok"))
    sp = DepthSpline(f, 10.0, 15.0)
    for D in rng.uniform(10.0, 15.0, 5):
        assert np.abs(sp(D) - f(D)).max() < 1e-5
    assert np.allclose(sp(sp.grid[3]), f(sp.grid[3]), atol=1e-12)


# ------------------------------------------------------------------------------------------
# ortak ters çözüm (3B)
# ------------------------------------------------------------------------------------------
GEOM = [trace_pieces(np.linspace(-55, 30, 18), 3.0 + 0.03 * np.linspace(-55, 30, 18), ext=100.0),
        trace_pieces(np.linspace(-12, 50, 13), -2.0 - 0.05 * np.clip(np.linspace(-12, 50, 13) - 25, 0, None), ext=100.0)]
KAPPA = -0.24


def synthetic3d(V=(2.0, 1.0), D=12.0, variant="blok", ng=45, ni=400, seed=0, gps_sd=0.5, sig=1.9, lam=20.0, nug=1.0):
    rng = np.random.default_rng(seed)
    xg, yg = rng.uniform(-50, 50, ng), rng.uniform(-50, 50, ng)
    xi, yi = rng.uniform(-50, 50, ni), rng.uniform(-50, 50, ni)
    x, y = np.r_[xg, xi], np.r_[yg, yi]
    Ux, Uy, Uz = make_green(x, y, GEOM, variant)(D)
    vx, vy = Ux @ np.array(V), Uy @ np.array(V)
    gps = GPSSet(y=yg, x=xg, vpar=vx[:ng] + 0.7 + rng.normal(0, gps_sd, ng),
                 vperp=vy[:ng] - 0.4 + 0.01 * yg + rng.normal(0, gps_sd, ng),
                 spar=np.full(ng, 0.05), sperp=np.full(ng, 0.05))
    C0 = exp_cov(xi, yi, sig, lam, nug)
    h = vx[ng:] + KAPPA * (vy[ng:] + 0.01 * yi) + 0.5 + 0.01 * xi - 0.02 * yi + np.linalg.cholesky(C0) @ rng.normal(size=ni)
    tr = InSARSet(name="yatay", x=xi, y=yi, los=h, pE=np.ones(ni), pN=np.full(ni, KAPPA), pU=np.zeros(ni), cov0=C0)
    return gps, [tr], x, y


def _model(gps, tracks, x, y, variant="blok", bounds=None):
    return JointModel3D(S, N, gps, tracks, green=make_green(x, y, GEOM, variant), fault_names=["A", "B"],
                        bounds=bounds or dict(D=(10.0, 15.0)))


def test_model3d_names_and_marginal_matches_dense():
    gps, tracks, x, y = synthetic3d(ng=8, ni=15, seed=4)
    m = _model(gps, tracks, x, y)
    assert m.theta_names == ["D", "log_floor_par", "log_floor_perp", "log_s_yatay"]
    assert [n for n in m.lin_names if n.startswith("V_")] == ["V_1", "V_2"]
    theta = np.array([m._spl.grid[5], np.log(0.6), np.log(1.4), np.log(0.9)])
    G, d, C, Psd = m.dense(theta)
    ref = multivariate_normal(mean=np.zeros(len(d)), cov=C + G @ np.diag(Psd ** 2) @ G.T).logpdf(d)
    assert m.log_marginal(theta) == pytest.approx(ref, rel=1e-9, abs=1e-6)
    assert not np.isfinite(m.log_post(np.r_[9.0, theta[1:]]))         # D sınır dışı


def test_model3d_track_columns_are_projections():
    """İz sütunu = pE·ux' + pN·uy' + pU·uz (p doğu/kuzey bileşenleri; profil 90° değilse dönüşüm gerekir).
    Bağımsız sınama: GPS istasyonlarıyla aynı noktalarda p = s ise GPS paralel, p = n ise GPS dik, p = yukarı
    ise uz sütunu çıkmalı. Profil doğrultusu 63° (s, n doğu/kuzeyden farklı)."""
    a = np.deg2rad(63.0); s, n = np.array([np.sin(a), np.cos(a)]), np.array([-np.cos(a), np.sin(a)])
    rng = np.random.default_rng(11)
    xg, yg = rng.uniform(-40, 40, 25), rng.uniform(-30, 30, 25)
    gps = GPSSet(y=yg, x=xg, vpar=np.zeros(25), vperp=np.zeros(25), spar=np.ones(25), sperp=np.ones(25))
    one, zero = np.ones(25), np.zeros(25)
    mk = lambda nm, pe, pn, pu: InSARSet(name=nm, x=xg, y=yg, los=zero, pE=pe * one, pN=pn * one, pU=pu * one, cov0=np.eye(25))
    tracks = [mk("s", *s, 0.0), mk("n", *n, 0.0), mk("u", 0.0, 0.0, 1.0)]
    geom = [trace_pieces(np.array([-15.0, 0.0, 15.0]), np.array([0.0, 3.0, 1.0]), ext=20.0)]
    x, y = np.r_[xg, xg, xg, xg], np.r_[yg, yg, yg, yg]
    g = make_green(x, y, geom, "blok_krip")
    m = JointModel3D(s, n, gps, tracks, green=g, fault_names=["A"], bounds=dict(D=(10.0, 15.0)))
    Vg, Vt = m._fault_columns(np.array([12.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    Ux, Uy, Uz = g(12.0)
    assert np.abs(Uy[:25]).max() > 0.02 and np.abs(Uz[:25]).max() > 0.005     # sınama anlamlı: sıfır değil
    assert np.allclose(Vt[0], Vg[:25]) and np.allclose(Vt[1], Vg[25:]) and np.allclose(Vt[2], Uz[:25])
    assert np.allclose(Vg[:25], Ux[:25]) and np.allclose(Vg[25:], Uy[:25])


def test_model3d_gps_floors_order():
    """theta'daki log_floor_par yalnız paralel, log_floor_perp yalnız dik bileşenin varyansına eklenir."""
    gps, tracks, x, y = synthetic3d(ng=6, ni=10, seed=8)
    m = _model(gps, tracks, x, y)
    _, _, C, _ = m.dense(np.array([12.0, np.log(0.3), np.log(2.0), 0.0]))
    d = np.diag(C)
    assert np.allclose(d[:6], gps.spar ** 2 + 0.3 ** 2) and np.allclose(d[6:12], gps.sperp ** 2 + 2.0 ** 2)


def test_model3d_reduces_to_2d_for_long_straight_fault():
    """Tek, düz, sonsuz uzatmalı iz (y = 2): 3B model sütunları ve marjinal olabilirlik 2B modelle aynı."""
    rng = np.random.default_rng(7)
    ng, ni = 30, 120
    xg, yg, xi, yi = (rng.uniform(-50, 50, k) for k in (ng, ng, ni, ni))
    gps = GPSSet(y=yg, x=xg, vpar=screw(yg, 2.0, 12.0, 2.0) + rng.normal(0, .5, ng), vperp=rng.normal(0, .5, ng),
                 spar=np.full(ng, .1), sperp=np.full(ng, .1))
    C0 = exp_cov(xi, yi, 1.5, 10.0, 0.5)
    tr = InSARSet(name="yatay", x=xi, y=yi, los=screw(yi, 2.0, 12.0, 2.0) + rng.normal(0, 1, ni), pE=np.ones(ni),
                  pN=np.full(ni, KAPPA), pU=np.zeros(ni), cov0=C0)
    geom = [trace_pieces(np.array([-50.0, 0.0, 50.0]), np.full(3, 2.0), ext=1e6)]
    x, y = np.r_[xg, xi], np.r_[yg, yi]
    m3 = JointModel3D(S, N, gps, [tr], green=make_green(x, y, geom, "blok"), fault_names=["A"], bounds=dict(D=(10.0, 15.0)))
    m2 = JointModel(S, N, gps, [tr], faults=[dict(y0=2.0, free=False)], knots=np.array([]), bounds=dict(D=(10.0, 15.0)))
    for D in (10.3, 12.0, 14.6):
        hyp = [np.log(0.5), np.log(0.5), 0.0]
        Vg3, Vt3 = m3._fault_columns(np.r_[D, hyp]); Vg2, Vt2 = m2._fault_columns(np.r_[D, hyp])
        assert np.abs(Vg3 - Vg2).max() < 3e-4 and np.abs(Vt3[0] - Vt2[0]).max() < 3e-4
        assert m3.log_marginal(np.r_[D, hyp]) == pytest.approx(m2.log_marginal(np.r_[D, hyp]), abs=0.02)


@pytest.mark.parametrize("variant", ["blok", "derin"])
def test_model3d_recovers_two_fault_rates(variant):
    """Gerçekçi gürültüde: tek tek V'ler ve toplam 3σ içinde; ~5 km aralıklı iki fayda V'ler güçlü ters
    ilişkili, TOPLAM tek tek V'lerden çok daha iyi belirlenir."""
    V = (2.0, 1.0)
    gps, tracks, x, y = synthetic3d(V=V, D=12.0, variant=variant, seed=1)
    m = _model(gps, tracks, x, y, variant=variant)
    th = np.array([12.0, np.log(0.5), np.log(0.5), 0.0])
    mean, cov = m.conditional(th)
    sd = np.sqrt(np.diag(cov)[:2])
    for k in range(2):
        assert abs(mean[k] - V[k]) < 3 * sd[k]
    s = mean[0] + mean[1]; sd_s = np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
    assert abs(s - sum(V)) < 3 * sd_s
    assert cov[0, 1] / (sd[0] * sd[1]) < -0.8 and sd_s < sd.min()


@pytest.mark.parametrize("variant", ["blok", "derin"])
def test_model3d_posterior_is_calibrated(variant):
    """20 sentetik veri seti (düşük gürültü, doğru hiperparametreler): z = (tahmin − gerçek)/σ için
    ortalama ~0 ve standart sapma ~1 (koşullu belirsizlikler ne iyimser ne kötümser)."""
    V = (2.0, 1.0); z = []
    for seed in range(10, 30):
        gps, tracks, x, y = synthetic3d(V=V, D=12.0, variant=variant, seed=seed, gps_sd=0.15, sig=0.5, nug=0.2)
        mean, cov = _model(gps, tracks, x, y, variant=variant).conditional(np.array([12.0, np.log(.15), np.log(.15), 0.0]))
        sd_s = np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
        z.append([(mean[0] - V[0]) / np.sqrt(cov[0, 0]), (mean[1] - V[1]) / np.sqrt(cov[1, 1]), (mean[0] + mean[1] - 3.0) / sd_s])
    z = np.array(z)
    assert np.all(np.abs(z.mean(0)) < 0.6) and np.all((z.std(0) > 0.6) & (z.std(0) < 1.4))
    assert np.sqrt(cov[0, 0]) < 1.0                                    # düşük gürültüde tek tek V de çözülür


def test_model3d_wrong_variant_fits_worse():
    """derin veriyle: derin modelin marjinal olabilirliği blok(100 km)'dan yüksek (kurgular ayırt edilebilir)."""
    gps, tracks, x, y = synthetic3d(V=(3.0, 2.0), D=12.0, variant="derin", seed=2, gps_sd=0.3)
    th = np.array([12.0, np.log(0.3), np.log(0.3), 0.0])
    assert _model(gps, tracks, x, y, "derin").log_marginal(th) > _model(gps, tracks, x, y, "blok").log_marginal(th) + 3.0


def test_model3d_sampler_smoke():
    gps, tracks, x, y = synthetic3d(ng=15, ni=60, seed=5)
    m = _model(gps, tracks, x, y)
    out = sample(m, nwalkers=10, nsteps=30, burn=10, seed=0, ndraw_lin=40)
    assert out["theta"].shape[1] == 4 and np.all((out["theta"][:, 0] >= 10) & (out["theta"][:, 0] <= 15))
    assert np.all(np.isfinite(out["logp"]))


# ------------------------------------------------------------------------------------------
# bölgesel hız gradyanı / dönme (isteğe bağlı doğrusal nuisance)
# ------------------------------------------------------------------------------------------
def test_regional_gradient_columns_and_dense():
    """Lxx = dv_par/dx, Lxy = dv_par/dy, Lyx = dv_perp/dx, omega: rijit dönme (v_par = −ω y, v_perp = +ω x;
    x→y saat yönü tersi). İz satırları q·v. Gradyanlar lin_names'in SONUNA eklenir (eski indeksler korunur)."""
    gps, tracks, x, y = synthetic3d(ng=7, ni=12, seed=6)
    grads = ("Lxx", "Lyx", "Lxy", "omega")
    m0 = _model(gps, tracks, x, y)
    m = JointModel3D(S, N, gps, tracks, green=make_green(x, y, GEOM, "blok"), fault_names=["A", "B"],
                     bounds=dict(D=(10.0, 15.0)), gradients=grads)
    assert m.lin_names[:len(m0.lin_names)] == m0.lin_names and m.lin_names[len(m0.lin_names):] == list(grads)
    theta = np.array([m._spl.grid[2], np.log(0.6), np.log(1.4), np.log(0.9)])
    G, d, C, Psd = m.dense(theta)
    ng = 7; i = {g: m.lin_names.index(g) for g in grads}
    xg, yg = gps.x, gps.y
    assert np.allclose(G[:ng, i["Lxx"]], xg) and np.allclose(G[ng:2 * ng, i["Lxx"]], 0)
    assert np.allclose(G[:ng, i["Lxy"]], yg) and np.allclose(G[ng:2 * ng, i["Lxy"]], 0)
    assert np.allclose(G[:ng, i["Lyx"]], 0) and np.allclose(G[ng:2 * ng, i["Lyx"]], xg)
    assert np.allclose(G[:ng, i["omega"]], -yg) and np.allclose(G[ng:2 * ng, i["omega"]], xg)
    tr = tracks[0]; qp, qn = tr.pE, tr.pN                                 # S = doğu, N = kuzey
    Gt = G[2 * ng:]
    assert np.allclose(Gt[:, i["Lxx"]], qp * tr.x) and np.allclose(Gt[:, i["Lyx"]], qn * tr.x)
    assert np.allclose(Gt[:, i["Lxy"]], qp * tr.y) and np.allclose(Gt[:, i["omega"]], -qp * tr.y + qn * tr.x)
    ref = multivariate_normal(mean=np.zeros(len(d)), cov=C + G @ np.diag(Psd ** 2) @ G.T).logpdf(d)
    assert m.log_marginal(theta) == pytest.approx(ref, rel=1e-9, abs=1e-6)
    # önsel: sıfır ortalamalı, sd = 0.2 mm/yıl/km (≈ ±10 mm/yıl 50 km'de; ~11°/Myr dönme), değiştirilebilir
    assert np.allclose(Psd[[i[g] for g in grads]], 0.2)
    m5 = JointModel3D(S, N, gps, tracks, green=make_green(x, y, GEOM, "blok"), fault_names=["A", "B"],
                      bounds=dict(D=(10.0, 15.0)), gradients=grads, prior_sd=dict(grad=0.05))
    assert np.allclose(m5.prior_sd[[i[g] for g in grads]], 0.05)


def test_regional_gradients_in_2d_model_and_requires_x():
    gps, tracks, x, y = synthetic3d(ng=7, ni=12, seed=6)
    m2 = JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=False)], knots=np.array([]), gradients=("Lyx", "omega"))
    G, d, C, Psd = m2.dense(np.array([12.0, np.log(.5), np.log(.5), 0.0]))
    assert np.allclose(G[7:14, m2.lin_names.index("Lyx")], gps.x)
    ref = multivariate_normal(mean=np.zeros(len(d)), cov=C + G @ np.diag(Psd ** 2) @ G.T).logpdf(d)
    assert m2.log_marginal(np.array([12.0, np.log(.5), np.log(.5), 0.0])) == pytest.approx(ref, rel=1e-9, abs=1e-6)
    gps.x = None
    with pytest.raises(ValueError):
        JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=False)], knots=np.array([]), gradients=("Lxx",))
    with pytest.raises(ValueError):
        JointModel(S, N, gps, tracks, faults=[dict(y0=1.0, free=False)], knots=np.array([]), gradients=("Lyy",))


def test_regional_gradients_recovered_with_fault():
    """Fay + dönme + Lxx + Lyx ile üretilmiş veri: gradyanlı model V toplamını ve gradyanları 3σ içinde geri
    alır; gradyansız model aynı veride belirgin biçimde daha kötü uyar."""
    V, om, lxx, lyx = (2.0, 1.0), 0.02, 0.01, -0.015
    gps, tracks, x, y = synthetic3d(V=V, seed=9, gps_sd=0.3)
    tr = tracks[0]
    gps.vpar = gps.vpar + lxx * gps.x - om * gps.y
    gps.vperp = gps.vperp + lyx * gps.x + om * gps.x
    tr.los = tr.los + tr.pE * (lxx * tr.x - om * tr.y) + tr.pN * (lyx * tr.x + om * tr.x)
    th = np.array([12.0, np.log(0.3), np.log(0.3), 0.0])
    g = make_green(x, y, GEOM, "blok")
    m = JointModel3D(S, N, gps, tracks, green=g, fault_names=["A", "B"], bounds=dict(D=(10.0, 15.0)), gradients=("Lxx", "Lyx", "omega"))
    mean, cov = m.conditional(th)
    for nm, true in (("Lxx", lxx), ("Lyx", lyx), ("omega", om)):
        j = m.lin_names.index(nm); assert abs(mean[j] - true) < 3 * np.sqrt(cov[j, j])
    sd_s = np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
    assert abs(mean[0] + mean[1] - 3.0) < 3 * sd_s
    m0 = JointModel3D(S, N, gps, tracks, green=g, fault_names=["A", "B"], bounds=dict(D=(10.0, 15.0)))
    assert m.log_marginal(th) > m0.log_marginal(th) + 10.0
