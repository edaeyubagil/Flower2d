"""3B Okada (1985) dikdörtgen dislokasyonu: yüzey yer değiştirmesi / hızı.

Uzlaşımlar: ss > 0 = SAĞ-YANAL (forward.py'deki V > 0 ile aynı), ds > 0 = ters/bindirme
(tavan bloğu yukarı); strike kuzeyden saat yönünde, fay doğrultunun SAĞINA eğimli.

Doğruluk, kapalı-biçim formüllerden BAĞIMSIZ kontrollerle kilitlenir:
  1. 2B limit: uzun düşey fay -> vida dislokasyonu (V/pi) atan(y/D)  [forward.screw]
  2. Süperpozisyon / geri-kayma özdeşliği
  3. Nokta kaynak, Boussinesq (dik) + Cerruti (teğet) yüzey-yükü Green fonksiyonları ve Betti
     karşıtlığıyla kurulur (Okada'nın nokta-kaynak formülleri KULLANILMAZ); Green fonksiyonlarının
     kendisi de burada denge, serbest yüzey ve kuvvet dengesiyle doğrulanır. Dikdörtgen üzerinde
     Gauss-Legendre ile (yakınsayana dek inceltilerek) integrallenip kapalı biçimle karşılaştırılır.
  4. Simetriler (yansıma, dönme), 5. işaret, + yüzeyi kesen fayda iz boyunca sıçrama = kayma vektörü.
"""
import numpy as np
import pytest

from profil2d.forward import Fault, screw, enu_velocity
from profil2d.geometry import Profile
from profil2d.okada import okada_rect


def _unit(strike):
    a = np.deg2rad(strike)
    return np.array([np.sin(a), np.cos(a)]), np.array([-np.cos(a), np.sin(a)])   # s, n (sol)


def _points(ref, strike, xs, yn):
    """Fay çerçevesi (xs: s yönü, yn: sol-normal n yönü; ref'e göre, km) -> (doğu, kuzey)."""
    s, n = _unit(strike)
    xs, yn = np.asarray(xs, float), np.asarray(yn, float)
    return ref[0] + xs * s[0] + yn * n[0], ref[1] + xs * s[1] + yn * n[1]


def _par_perp(strike, ue, un):
    s, n = _unit(strike)
    return ue * s[0] + un * s[1], ue * n[0] + un * n[1]


# --------------------------------------------------------------------------------------------
# 1. 2B limit
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [0.25, 0.40])
@pytest.mark.parametrize("D", [3.0, 10.0, 25.0])
def test_long_vertical_fault_matches_2d_screw_dislocation(D, nu):
    """Uzun düşey fay, D'den 1e5 km'ye V hızıyla sağ-yanal kayıyor: orta noktadan geçen dik
    profilde faya paralel bileşen = (V/pi) atan(y/D), y SOL tarafa doğru (sol taraf +s gider).
    2B anti-düzlem çözümü Poisson oranından bağımsızdır; faya dik ve düşey bileşen ~0.
    Uzunluk 1e6 km: sonlu uzunluk hatası ~V·y/L (bkz. bir sonraki test); 1e5 km tabanın
    kesme hatası (V/pi)·y/1e5 <= 2.4e-4 V (y <= 75 km)."""
    V, strike, ref = 4.0, 63.0, (12.0, -7.0)
    y = np.array([-3, -1, -0.3, -0.1, -0.01, 0.01, 0.1, 0.3, 1, 3]) * D
    e, n = _points(ref, strike, np.zeros_like(y), y)
    ue, un, uu = okada_rect(e, n, *ref, strike, length=1e6, top=D, bottom=1e5, ss=V, nu=nu)
    vpar, vperp = _par_perp(strike, ue, un)
    assert np.allclose(vpar, screw(y, 0.0, D, V), rtol=0, atol=1e-3 * V)
    assert np.allclose(vperp, 0, atol=1e-6 * V)
    assert np.allclose(uu, 0, atol=1e-6 * V)


def test_2d_misfit_is_finite_length_effect():
    """L = 1e4 km'de kalan fark formül hatası değil, sonlu uzunluk etkisi: ~1/L ile azalır."""
    V, D = 4.0, 10.0
    y = np.array([-30.0, -10.0, -3.0, 3.0, 10.0, 30.0])
    misfit = []
    for L in (1e4, 1e5):
        ue, un, uu = okada_rect(-y, 0 * y, 0.0, 0.0, 0.0, length=L, top=D, bottom=1e8, ss=V)   # strike 0: n = batı
        misfit.append(np.abs(un - screw(y, 0.0, D, V)).max())
    assert 1e-3 * V < misfit[0] < 1e-2 * V
    assert misfit[0] / misfit[1] == pytest.approx(10.0, rel=0.1)


def test_long_fault_agrees_with_package_2d_model_on_profile():
    """profil2d.forward.enu_velocity (Profile, lon/lat) ile aynı yerel E,N alanı."""
    p = Profile(27.6, 39.6, 77.0)
    flt = Fault(y0=4.0, D=12.0, V=3.0)
    lon, lat = np.meshgrid(np.linspace(27.3, 27.9, 7), np.linspace(39.4, 39.8, 6))
    lon, lat = lon.ravel(), lat.ravel()
    ve, vn = enu_velocity(p, lon, lat, [flt])
    e, n = p.to_local(lon, lat)
    ref_e, ref_n = flt.y0 * p.n                       # fay izi: y = y0 doğrusu
    ue, un, uu = okada_rect(e, n, ref_e, ref_n, p.strike, length=1e6, top=flt.D, bottom=1e5, ss=flt.V)
    assert np.allclose(ue, ve, atol=1e-3 * flt.V) and np.allclose(un, vn, atol=1e-3 * flt.V)
    assert np.allclose(uu, 0, atol=1e-3 * flt.V)


# --------------------------------------------------------------------------------------------
# 2. Süperpozisyon
# --------------------------------------------------------------------------------------------
def _random_points(seed, k=300, half=60.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, k), rng.uniform(-half, half, k)


@pytest.mark.parametrize("dip", [90.0, 55.0])
def test_backslip_identity(dip):
    """[0, inf) kayması − [0, D] kayması = [D, inf) kayması. Derin parçanın üst kenarı D/tan(dip)
    kadar SAĞA (−n) kayık izdüşer (ref = üst kenar ortasının yüzey izdüşümü)."""
    e, n = _random_points(1)
    strike, D, big, ref = 123.0, 11.0, 1e5, (3.0, -2.0)
    _, nv = _unit(strike)
    shift = D / np.tan(np.deg2rad(dip))
    ref_D = (ref[0] - shift * nv[0], ref[1] - shift * nv[1])
    kw = dict(strike_deg=strike, length=40.0, dip_deg=dip, ss=2.5, ds=-1.3)
    full = okada_rect(e, n, *ref, top=0.0, bottom=big, **kw)
    shallow = okada_rect(e, n, *ref, top=0.0, bottom=D, **kw)
    deep = okada_rect(e, n, *ref_D, top=D, bottom=big, **kw)
    for a, b, c in zip(full, shallow, deep):
        assert np.allclose(a - b, c, rtol=0, atol=1e-9)


@pytest.mark.parametrize("dip", [90.0, 40.0])
def test_along_strike_split(dip):
    """Uzunluk ref etrafında ortalanmış: [−L/2, L/2] = [−L/2, 0] + [0, L/2]."""
    e, n = _random_points(2)
    strike, L, ref = 301.0, 36.0, (-4.0, 6.0)
    s, _ = _unit(strike)
    kw = dict(strike_deg=strike, top=0.5, bottom=14.0, dip_deg=dip, ss=1.1, ds=0.7)
    whole = okada_rect(e, n, *ref, length=L, **kw)
    a = okada_rect(e, n, ref[0] - L / 4 * s[0], ref[1] - L / 4 * s[1], length=L / 2, **kw)
    b = okada_rect(e, n, ref[0] + L / 4 * s[0], ref[1] + L / 4 * s[1], length=L / 2, **kw)
    for w, x, y in zip(whole, a, b):
        assert np.allclose(w, x + y, rtol=0, atol=1e-10)


# --------------------------------------------------------------------------------------------
# 3. Bağımsız nokta kaynak + sayısal integral
# --------------------------------------------------------------------------------------------
# Çerçeve: (doğu, kuzey, AŞAĞI), yarı uzay z >= 0, kayma modülü G = 1 (sonuçtan düşer).
def _surface_load_green(r, nu):
    """g[i, k] = yüzeyde orijine uygulanan birim kuvvet e_k'nin r = (x, y, z) noktasında
    oluşturduğu yer değiştirmenin i bileşeni. k = 0, 1: Cerruti (teğet); k = 2: Boussinesq
    (dik, katıya doğru = aşağı). Johnson (1985, Contact Mechanics, §2.2) biçimi; aşağıdaki testte
    denge + serbest yüzey + kuvvet dengesiyle doğrulanır. r karmaşık olabilir (karmaşık-adım)."""
    x, y, z = r
    rho = np.sqrt(x * x + y * y + z * z)
    a, c = 1.0 - 2.0 * nu, 1.0 / (4.0 * np.pi)
    r3, rz = rho**3, rho + z
    A, B = a / (rho * rz**2), a / (rho * rz)
    gxy = c * (x * y / r3 - x * y * A)
    return np.array([
        [c * (1 / rho + x * x / r3 + a / rz - x * x * A), gxy, c * (x * z / r3 - x * B)],
        [gxy, c * (1 / rho + y * y / r3 + a / rz - y * y * A), c * (y * z / r3 - y * B)],
        [c * (x * z / r3 + x * B), c * (y * z / r3 + y * B), c * (z * z / r3 + 2.0 * (1.0 - nu) / rho)]])


def _stress(r, nu, h=1e-30):
    """sig[i, j, k] = e_k kuvvetinin gerilmesi sigma_ij (G = 1); gradyan karmaşık-adım türeviyle."""
    r = np.asarray(r, float)
    grad = np.empty((3, 3, 3) + r.shape[1:])
    for j in range(3):
        rc = r.astype(complex)
        rc[j] = rc[j] + 1j * h
        grad[:, j] = _surface_load_green(rc, nu).imag / h               # du_i^(k) / dr_j
    eps = 0.5 * (grad + grad.swapaxes(0, 1))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2.0 * eps
    for i in range(3):
        sig[i, i] = sig[i, i] + 2.0 * nu / (1.0 - 2.0 * nu) * tr        # lambda / G
    return sig


@pytest.mark.parametrize("k", [0, 1, 2])
def test_surface_load_green_functions(k):
    """Green fonksiyonlarının kendisi: (a) denge div(sigma) = 0, (b) z = 0'da serbest yüzey
    sigma_iz = 0, (c) yarıküre üzerindeki çekiş toplamı = −f (birim kuvvet, doğru işaret)."""
    nu, f = 0.3, np.eye(3)[k]
    r = np.array([[1.3, -0.4, 0.9], [-2.0, 1.1, 0.3], [0.2, 0.5, 2.5], [0.7, 0.6, 0.05]]).T
    h = 1e-4
    div = sum((_stress(r + h * np.eye(3)[:, [j]], nu)[:, j, k] - _stress(r - h * np.eye(3)[:, [j]], nu)[:, j, k])
              / (2 * h) for j in range(3))
    scale = np.abs(_stress(r, nu)[:, :, k]).max(axis=(0, 1)) / np.linalg.norm(r, axis=0)
    assert np.all(np.abs(div).max(axis=0) < 1e-6 * scale)

    rs = np.array([[1.0, 0.3, 0.0], [-0.7, 2.0, 0.0], [3.0, -1.0, 0.0], [0.0, -0.4, 0.0]]).T
    sig = _stress(rs, nu)[:, :, k]
    assert np.abs(sig[:, 2]).max() < 1e-12 * np.abs(sig).max()

    a = 2.0
    th, wt = np.polynomial.legendre.leggauss(48)
    th, wt = np.pi / 4 * (th + 1), np.pi / 4 * wt                  # kutup açısı, aşağı eksenden
    ph = np.linspace(0, 2 * np.pi, 96, endpoint=False); wp = 2 * np.pi / 96
    T, P = np.meshgrid(th, ph, indexing="ij")
    nrm = np.array([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)])
    traction = np.einsum("ij...,j...->i...", _stress(a * nrm, nu)[:, :, k], nrm)
    total = np.einsum("itp,t->i", traction * a * a * np.sin(T), wt) * wp
    assert np.allclose(total, -f, atol=1e-10)


def _gauss_legendre(a, b, npanel, order=10):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, npanel + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    return (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel(), (0.5 * (hi - lo) * w + 0 * lo).ravel()


def _fault_geometry_END(strike, dip):
    """(doğu, kuzey, aşağı) çerçevesinde s, eğim-aşağı birim vektör, taban->tavan normali."""
    a, d = np.deg2rad(strike), np.deg2rad(dip)
    s = np.array([np.sin(a), np.cos(a), 0.0])
    right = np.array([np.cos(a), -np.sin(a), 0.0])                 # doğrultunun sağı = eğim yönü
    down = np.array([0.0, 0.0, 1.0])
    return s, np.cos(d) * right + np.sin(d) * down, np.sin(d) * right - np.cos(d) * down


def _point_source_integral(e, n, ref, strike, length, top, bottom, dip, ss, ds, nu, h):
    """Betti karşıtlığı: u_k(X) = ∫ [u]_i nu_j sigma^(k)_ij(xi − X) dSigma, sigma^(k) = X'teki birim
    yüzey kuvveti e_k'nin xi'deki gerilmesi; [u] = u(tavan) − u(taban), nu taban->tavan normali."""
    s, dd, nrm = _fault_geometry_END(strike, dip)
    slip = -ss * s - ds * dd           # sağ-yanal: tavan (sağ) blok −s; ters: tavan blok yukarı-eğim
    W = (bottom - top) / np.sin(np.deg2rad(dip))
    al, wa = _gauss_legendre(-length / 2, length / 2, int(np.ceil(length / h)))
    be, wb = _gauss_legendre(0.0, W, int(np.ceil(W / h)))
    P = np.array([ref[0], ref[1], top]) + al[:, None, None] * s + be[None, :, None] * dd
    P, w = P.reshape(-1, 3).T, (wa[:, None] * wb[None, :]).ravel()
    out = np.zeros((3, np.size(e)))
    for m, (xe, xn) in enumerate(zip(np.ravel(e), np.ravel(n))):
        rel = P - np.array([[xe], [xn], [0.0]])
        out[:, m] = np.einsum("i,j,ijkm,m->k", slip, nrm, _stress(rel, nu), w)
    return out[0], out[1], -out[2]                                   # yukarı = −aşağı


def _distance_to_fault(e, n, ref, strike, length, top, bottom, dip):
    s, dd, _ = _fault_geometry_END(strike, dip)
    W = (bottom - top) / np.sin(np.deg2rad(dip))
    rel = np.stack([e - ref[0], n - ref[1], -top + 0 * e])
    a = np.clip(np.einsum("i,i...->...", s, rel), -length / 2, length / 2)
    b = np.clip(np.einsum("i,i...->...", dd, rel), 0.0, W)
    return np.linalg.norm(rel - a * s[:, None] - b * dd[:, None], axis=0)


POINT_SOURCE_CASES = [
    # strike, dip, top, bottom, length, ss, ds, nu
    (37.0, 90.0, 0.0, 12.0, 30.0, 1.0, 0.0, 0.25),
    (145.0, 90.0, 2.0, 15.0, 24.0, 0.0, 1.0, 0.25),
    (250.0, 60.0, 0.0, 14.0, 30.0, 1.0, 0.0, 0.30),
    (300.0, 60.0, 3.0, 18.0, 26.0, 0.0, 1.0, 0.25),
    (10.0, 30.0, 1.0, 10.0, 20.0, 0.7, -1.2, 0.25),
    (200.0, 15.0, 0.0, 6.0, 20.0, 0.0, 1.0, 0.25),     # sığ eğim: tavan bloğu üstünde ayrı dal
    (75.0, 89.9, 0.5, 10.0, 20.0, 1.0, 0.5, 0.20),     # neredeyse düşey: genel (cos dip != 0) dal
]


@pytest.mark.parametrize("strike, dip, top, bottom, length, ss, ds, nu", POINT_SOURCE_CASES)
def test_rectangle_matches_integrated_point_source(strike, dip, top, bottom, length, ss, ds, nu):
    ref = (1.5, -2.0)
    xs, yn = np.meshgrid([-30.0, -14.0, -3.0, 6.0, 17.0, 28.0], [-24.0, -9.0, -4.0, 3.5, 9.0, 20.0])
    xs, yn = list(xs.ravel()), list(yn.ravel())
    xs += [length / 2 + 4, -length / 2 - 4, 2.0]           # iz uzantısı (q = 0) ve ...
    yn += [0.0, 0.0, top / np.tan(np.deg2rad(dip))]       # ... gömülü fay düzleminin yüzey kesişimi
    e, n = _points(ref, strike, xs, yn)
    keep = _distance_to_fault(e, n, ref, strike, length, top, bottom, dip) > 2.0
    e, n = e[keep], n[keep]
    assert e.size >= 25

    args = (e, n, ref, strike, length, top, bottom, dip, ss, ds, nu)
    prev = None
    for h in (4.0, 2.0, 1.0, 0.5):                         # yakınsayana dek panelleri incelt
        cur = np.array(_point_source_integral(*args, h=h))
        if prev is not None and np.abs(cur - prev).max() < 1e-11 * np.abs(cur).max():
            break
        prev = cur
    else:
        pytest.fail("Gauss-Legendre integrali yakınsamadı")

    got = np.array(okada_rect(e, n, *ref, strike, length, top, bottom, dip_deg=dip, ss=ss, ds=ds, nu=nu))
    assert np.abs(got - cur).max() < 1e-10 * np.abs(cur).max()       # gözlenen: ~1e-15 (89.9°: ~1e-13)


# --------------------------------------------------------------------------------------------
# 4. Simetriler
# --------------------------------------------------------------------------------------------
def test_vertical_strike_slip_mirror_symmetries():
    """Düşey doğrultu atımlı fayda bir ayna yansıması sağ-yanalı sol-yanala çevirir:
    fay düzlemine göre (y -> −y): paralel ve düşey ters-simetrik, dik bileşen simetrik;
    orta düzleme göre (x -> −x): düşey ve dik bileşen ters-simetrik, paralel simetrik."""
    strike, ref = 40.0, (5.0, -3.0)
    kw = dict(strike_deg=strike, length=30.0, top=1.0, bottom=15.0, ss=2.0)
    rng = np.random.default_rng(7)
    xs, yn = rng.uniform(-40, 40, 60), rng.uniform(0.2, 40, 60)

    def field(x, y):
        ue, un, uu = okada_rect(*_points(ref, strike, x, y), *ref, **kw)
        return (*_par_perp(strike, ue, un), uu)

    par, perp, up = field(xs, yn)
    par_y, perp_y, up_y = field(xs, -yn)
    par_x, perp_x, up_x = field(-xs, yn)
    tol = dict(rtol=0, atol=1e-12)
    assert np.allclose(par_y, -par, **tol) and np.allclose(perp_y, perp, **tol) and np.allclose(up_y, -up, **tol)
    assert np.allclose(up_x, -up, **tol) and np.allclose(perp_x, -perp, **tol) and np.allclose(par_x, par, **tol)
    assert np.abs(up).max() > 1e-2 and np.abs(perp).max() > 1e-2      # test boş değil


@pytest.mark.parametrize("dip, ss, ds", [(90.0, 2.0, 0.0), (90.0, 0.0, 1.0), (50.0, 1.0, -1.5)])
@pytest.mark.parametrize("theta", [17.0, 90.0, 133.0, 241.0, -60.0])
def test_rotation_invariance(theta, dip, ss, ds):
    """Fay ve noktalar theta kadar (saat yönünde) döndürülünce vektör alanı da aynı dönüşü yapar."""
    e, n = _random_points(3, k=80, half=50.0)
    ref, strike = (4.0, -6.0), 25.0
    kw = dict(length=30.0, top=1.0, bottom=15.0, dip_deg=dip, ss=ss, ds=ds)
    ue, un, uu = okada_rect(e, n, *ref, strike, **kw)
    t = np.deg2rad(theta)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # azimut alfa -> alfa + theta
    e2, n2 = rot @ np.array([e, n])
    ue2, un2, uu2 = okada_rect(e2, n2, *(rot @ np.array(ref)), strike + theta, **kw)
    assert np.allclose(np.array([ue2, un2]), rot @ np.array([ue, un]), rtol=0, atol=1e-11)
    assert np.allclose(uu2, uu, rtol=0, atol=1e-11)


@pytest.mark.parametrize("strike", [0.0, 77.0, 200.0, -35.0])
def test_reversed_strike_of_vertical_fault(strike):
    """Düşey fay strike+180 ile yazılırsa aynı fiziksel fay: sağ-yanal yine sağ-yanaldır (aynı alan);
    ds'de 'tavan bloğu' (doğrultunun sağı) yer değiştirir, yani ds işaret değiştirir."""
    e, n = _random_points(4, k=80, half=50.0)
    kw = dict(ref_e=1.0, ref_n=2.0, length=25.0, top=2.0, bottom=14.0)
    for a, b in zip(okada_rect(e, n, strike_deg=strike, ss=1.7, **kw),
                    okada_rect(e, n, strike_deg=strike + 180.0, ss=1.7, **kw)):
        assert np.allclose(a, b, rtol=0, atol=1e-12)
    for a, b in zip(okada_rect(e, n, strike_deg=strike, ds=0.9, **kw),
                    okada_rect(e, n, strike_deg=strike + 180.0, ds=-0.9, **kw)):
        assert np.allclose(a, b, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------------
# 5. İşaret ve iz boyunca sıçrama
# --------------------------------------------------------------------------------------------
def test_right_lateral_sign_east_west_fault():
    """D–B uzanımlı (strike 90) sağ-yanal fay: kuzey noktaları güneye göre DOĞUYA gider."""
    e = np.array([0.0, 0.0]); n = np.array([-40.0, 40.0])            # güney, kuzey
    ue, un, uu = okada_rect(e, n, 0.0, 0.0, 90.0, length=1e6, top=10.0, bottom=1e5, ss=5.0)
    assert ue[1] > 0 > ue[0]
    assert ue[1] - ue[0] == pytest.approx(2 * 5.0 / np.pi * np.arctan(4.0), abs=5e-3)
    assert np.allclose(un, 0, atol=1e-9) and np.allclose(uu, 0, atol=1e-9)


@pytest.mark.parametrize("dip", [90.0, 60.0, 30.0, 12.0])
def test_surface_breaking_fault_offset_equals_slip(dip):
    """Yüzeyi kesen fayda iz boyunca süreksizlik = kayma vektörü:
    u(sağ/tavan) − u(sol/taban) = −ss·s + ds·(cos(dip)·n, sin(dip)·yukarı)."""
    strike, ref, ss, ds = 110.0, (2.0, 3.0), 1.7, -0.8
    s, nv = _unit(strike)
    xs, eps = np.array([-5.0, 0.0, 8.0]), 1e-7
    kw = dict(ref_e=ref[0], ref_n=ref[1], strike_deg=strike, length=30.0, top=0.0, bottom=12.0,
              dip_deg=dip, ss=ss, ds=ds)
    left = np.array(okada_rect(*_points(ref, strike, xs, +eps + 0 * xs), **kw))
    right = np.array(okada_rect(*_points(ref, strike, xs, -eps + 0 * xs), **kw))
    c, sd = np.cos(np.deg2rad(dip)), np.sin(np.deg2rad(dip))
    expected = np.array([-ss * s[0] + ds * c * nv[0], -ss * s[1] + ds * c * nv[1], ds * sd])
    assert np.allclose(right - left, expected[:, None], rtol=0, atol=1e-5)


def test_on_trace_strike_slip_is_mean_of_both_sides():
    """Tam iz üzerinde (q = 0) doğrultu atımlı çözüm sonludur ve iki tarafın ortalamasıdır."""
    strike, ref = 33.0, (0.0, 0.0)
    xs = np.array([-9.0, 0.0, 4.0])
    kw = dict(ref_e=0.0, ref_n=0.0, strike_deg=strike, length=30.0, top=0.0, bottom=10.0, ss=2.0)
    on = np.array(okada_rect(*_points(ref, strike, xs, 0 * xs), **kw))
    lft = np.array(okada_rect(*_points(ref, strike, xs, 0 * xs + 1e-7), **kw))
    rgt = np.array(okada_rect(*_points(ref, strike, xs, 0 * xs - 1e-7), **kw))
    assert np.all(np.isfinite(on))
    assert np.allclose(on, 0.5 * (lft + rgt), rtol=0, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Sayısal kararlılık, şekiller, girdi denetimi
# --------------------------------------------------------------------------------------------
def test_dip_continuity_near_vertical():
    """dip -> 90: genel dal (1/cos dip terimleri) düzgün biçimde düşey dala yakınsar."""
    e, n = _random_points(5, k=200, half=40.0)
    kw = dict(ref_e=0.5, ref_n=-1.0, strike_deg=70.0, length=30.0, top=0.0, bottom=15.0, ss=1.0, ds=1.0)
    u90 = np.array(okada_rect(e, n, dip_deg=90.0, **kw))
    for eps in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9):
        diff = np.abs(np.array(okada_rect(e, n, dip_deg=90.0 - eps, **kw)) - u90).max()
        assert diff < 0.05 * eps + 1e-9, (eps, diff)


def test_shapes_and_scalars():
    e, n = np.meshgrid(np.linspace(-20, 20, 4), np.linspace(-10, 10, 3))
    out = okada_rect(e, n, 0.0, 0.0, 45.0, 20.0, 0.0, 10.0, ss=1.0)
    assert all(o.shape == (3, 4) for o in out)
    one = okada_rect(20.0, 10.0, 0.0, 0.0, 45.0, 20.0, 0.0, 10.0, ss=1.0)    # = ızgara [2, 3]
    assert np.allclose([float(o) for o in one], [o[2, 3] for o in out])


@pytest.mark.parametrize("bad", [dict(top=5.0, bottom=5.0), dict(top=-1.0), dict(length=0.0),
                                 dict(dip_deg=0.0), dict(dip_deg=95.0), dict(nu=0.5)])
def test_invalid_input_raises(bad):
    kw = dict(ref_e=0.0, ref_n=0.0, strike_deg=0.0, length=10.0, top=0.0, bottom=10.0, ss=1.0)
    kw.update(bad)
    with pytest.raises(ValueError):
        okada_rect(np.array([1.0]), np.array([2.0]), **kw)
