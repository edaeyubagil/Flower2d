"""C — 3B segment modeli: haritalı fay geometrisi + Okada (1985) dislokasyonları, ortak D, fay başına V.

Çerçeve: profil çerçevesi (x = s doğrultu boyunca, y = n sol-normal; km). (x, y), yerel (doğu, kuzey)
çerçevesinin dönmüş hâlidir ve aynı el-yönelimine sahiptir (x, y'nin 90° saat yönündedir); bu yüzden
okada_rect doğrudan (e = x, n = y) ile çağrılır. Parça doğrultusu buna göre atan2(dx, dy)'dir ve çıktı
(ux, uy) = (faya paralel [s], faya dik [n]) bileşenlerdir. V > 0 SAĞ-YANAL (forward / okada ile aynı).

Geometri
--------
merged_trace : bir fayın tüm segment izleri (bindirmeli paralel kollar dahil) x boyunca `step` km
               aralıklı düğümlerde YEREL DOĞRUSAL (ağırlıklı) uyumla tek bir ize indirgenir. Paralel kollar
               (ör. 12-4 / 12-5, ~4 km aralık) ortalanır, kısa boşluklar köprülenir. Gerekçe: D ≥ 10 km
               olduğundan kilitli bölgenin altındaki kayma, D'den küçük ölçekli yüzey ayrıntısını görmez;
               ayrı kollara aynı V'yi vermek ise toplam göreli hareketi iki katına çıkarırdı.
trace_pieces : düğümler arası düz parçalar (kind 'iz') + iki uçtan, zonun genel doğrultusunda (profil
               x ekseni; isteğe bağlı kiriş) `ext` km'lik uzatmalar (kind 'uzanti').

Kurgular (V = 1)
----------------
derin     : yalnız haritalı iz boyunca, D'den BIG'e kayma. Bölgesel hareket fay uçlarının ötesine taşınmaz.
blok      : iz + uzatmalar boyunca D'den BIG'e kayma. Doğrusallıkla tam olarak
            [iz+uzatma, 0..BIG blok hareketi] − [iz+uzatma, 0..D geri kayma] (Savage, 1983) ile aynıdır;
            uzatma çok uzunsa (≥ 1e5 km) kutu içinde katı blok sınırının etkisi, 2B limit (testli).
blok_krip : [iz+uzatma, 0..BIG] − [yalnız iz, 0..D]: haritalı iz kilitli, uzatmalar yüzeye kadar KRİPLİ
            (= blok + uzatmalarda 0..D kayma). Uç ötesinde yüzeyde keskin basamak öngörür.

Ters çözüm: JointModel3D, inversion.JointModel'in tüm nuisance terimlerini (a_par, a_perp, b_perp·y,
iz rampaları, düşey şapka tabanı) ve gürültü modelini (GPS taban σ'ları, InSAR ölçeği) aynen kullanır;
yalnız fay sütunları Okada'dan gelir ve D ortak TEK parametredir. D'ye bağlı (beyazlatılmış) sütunlar
log D ızgarasında kübik spline ile önceden hesaplanır (spline doğrusal olduğundan L⁻¹ ile sırası değişir).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular

from .okada import okada_rect
from .inversion import JointModel

BIG = 1e5        # "sonsuz" taban derinliği (km); test_okada: kesme hatası (V/pi)·y/BIG
VARIANTS = ("derin", "blok", "blok_krip")


# ------------------------------------------------------------------------------------------
# geometri
# ------------------------------------------------------------------------------------------
def _densify(parts, ds=0.25):
    pts, w = [], []
    for c in parts:
        c = np.asarray(c, float)[:, :2]
        for p0, p1 in zip(c[:-1], c[1:]):
            L = float(np.hypot(*(p1 - p0)))
            if L == 0.0:
                continue
            m = max(1, int(np.ceil(L / ds)))
            t = (np.arange(m) + 0.5) / m
            pts.append(p0 + t[:, None] * (p1 - p0)); w.append(np.full(m, L / m))
    return np.vstack(pts), np.concatenate(w)


def merged_trace(parts, step=5.0, window=None, ds=0.25):
    """Bir fayın izleri (profil çerçevesinde (k, 2) x, y dizileri) -> birleşik iz düğümleri (xn, yn).

    Düğümler köşelerin x aralığını eşit böler (aralık ≤ step). Her düğümde |x − xj| ≤ window
    (varsayılan step/2, boşluk yoksa) içindeki noktalara uzunluk ağırlıklı doğrusal uyum yapılır ve
    xj'de değerlendirilir (düz izde tam). Noktası olmayan düğümler komşulardan doğrusal ara değerlenir."""
    P, W = _densify(parts, ds)
    V = np.vstack([np.asarray(c, float)[:, :2] for c in parts])
    x0, x1 = V[:, 0].min(), V[:, 0].max()
    nseg = max(1, int(np.ceil((x1 - x0) / step - 1e-9)))
    xn = np.linspace(x0, x1, nseg + 1)
    win = 0.5 * (x1 - x0) / nseg if window is None else float(window)
    yn = np.full(len(xn), np.nan)
    for j, xj in enumerate(xn):
        k = np.abs(P[:, 0] - xj) <= win + 1e-9
        if not k.any():
            continue
        w, xx, yy = W[k], P[k, 0], P[k, 1]
        xm, ym = np.average(xx, weights=w), np.average(yy, weights=w)
        sxx = np.sum(w * (xx - xm) ** 2)
        slope = np.sum(w * (xx - xm) * (yy - ym)) / sxx if sxx > 1e-12 * np.sum(w) else 0.0
        yn[j] = ym + slope * (xj - xm)
    ok = np.isfinite(yn)
    yn = np.interp(xn, xn[ok], yn[ok])
    return xn, yn


def _piece(x0, y0, x1, y1, kind):
    dx, dy = x1 - x0, y1 - y0
    return dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), xm=0.5 * (x0 + x1), ym=0.5 * (y0 + y1),
                strike=float(np.degrees(np.arctan2(dx, dy))), length=float(np.hypot(dx, dy)), kind=kind)


def trace_pieces(xn, yn, ext=0.0, direction=(1.0, 0.0)):
    """Düğümler -> (iz parçaları, uzatma parçaları).

    direction: uzatma yönü, profil çerçevesinde vektör; varsayılan (1, 0) = zonun genel doğrultusu (profil
    x ekseni; uzatmalar birbirine paralel, kesişmez). "kiris": ilk→son düğüm kirişi (bükük fayda, ör.
    Balıkesir'in Kepsut ucu, kiriş eğilir ve uzatma komşu fayı kesebilir)."""
    xn, yn = np.asarray(xn, float), np.asarray(yn, float)
    tr = [_piece(xn[i], yn[i], xn[i + 1], yn[i + 1], "iz") for i in range(len(xn) - 1)]
    ex = []
    if ext > 0:
        u = np.array([xn[-1] - xn[0], yn[-1] - yn[0]]) if isinstance(direction, str) and direction == "kiris" \
            else np.asarray(direction, float).copy()
        if u @ np.array([xn[-1] - xn[0], yn[-1] - yn[0]]) < 0:
            u = -u                                                       # iz yönüyle aynı yön
        u /= np.hypot(*u)
        ex = [_piece(xn[0] - ext * u[0], yn[0] - ext * u[1], xn[0], yn[0], "uzanti"),
              _piece(xn[-1], yn[-1], xn[-1] + ext * u[0], yn[-1] + ext * u[1], "uzanti")]
    return tr, ex


# ------------------------------------------------------------------------------------------
# Green fonksiyonları
# ------------------------------------------------------------------------------------------
def _okada_sum(x, y, pieces, top, bottom):
    out = np.zeros((3, np.size(x)))
    for p in pieces:
        out += np.array(okada_rect(x, y, p["xm"], p["ym"], p["strike"], p["length"], top, bottom, ss=1.0))
    return out


def fault_green(x, y, trace, ext, D, variant):
    """V = 1 (sağ-yanal) için yüzey hızı (ux [faya paralel], uy [faya dik], uz) — kurgular modül belgesinde."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if variant == "derin":
        u = _okada_sum(x, y, trace, D, BIG)
    elif variant == "blok":
        u = _okada_sum(x, y, list(trace) + list(ext), D, BIG)
    elif variant == "blok_krip":
        u = _okada_sum(x, y, list(trace) + list(ext), D, BIG) + _okada_sum(x, y, ext, 0.0, D)
    else:
        raise ValueError(f"bilinmeyen kurgu {variant!r}; seçenekler {VARIANTS}")
    return u[0], u[1], u[2]


def make_green(x, y, faults, variant):
    """faults: [(iz, uzatma) her fay için]. Dönüş: D -> (Ux, Uy, Uz), her biri (nokta, fay)."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()

    def g(D):
        cols = [fault_green(x, y, tr, ex, D, variant) for tr, ex in faults]
        return tuple(np.stack([c[i] for c in cols], axis=1) for i in range(3))
    return g


class DepthSpline:
    """f(D) (herhangi şekilli dizi) için log D ızgarasında kübik spline; ızgara aralığı ≤ dlog (oransal)."""

    def __init__(self, fun, Dmin, Dmax, dlog=0.02):
        n = max(4, int(np.ceil(np.log(Dmax / Dmin) / dlog)) + 1)
        self.grid = np.exp(np.linspace(np.log(Dmin), np.log(Dmax), n))
        self.grid[0], self.grid[-1] = Dmin, Dmax
        vals = np.stack([np.asarray(fun(D), float) for D in self.grid])
        self._cs = CubicSpline(np.log(self.grid), vals, axis=0)

    def __call__(self, D):
        return self._cs(np.log(D))


# ------------------------------------------------------------------------------------------
# ortak ters çözüm
# ------------------------------------------------------------------------------------------
class JointModel3D(JointModel):
    """GPS + InSAR veri setleri, 3B fay sütunları. green(D) -> (Ux, Uy, Uz) (nokta, fay); nokta sırası:
    önce GPS istasyonları (gps.y sırası), ardından her iz kendi sırasıyla.
    theta = [D, log σ_taban_par, log σ_taban_perp, log s_iz...];  doğrusal: [V_1..V_K, a_par, a_perp, b_perp,
    rampalar..., u...] (2B ile aynı adlar). D sınırları bounds['D'] (düzgün önsel)."""

    def __init__(self, s, n, gps, tracks, green, fault_names, knots=(), prior_sd=None, bounds=None, dlog=0.02, gradients=()):
        self.green, self.fault_names = green, list(fault_names)
        super().__init__(s, n, gps, tracks, faults=[dict(name=nm) for nm in self.fault_names],
                         knots=np.asarray(knots, float), prior_sd=prior_sd, bounds=bounds, gradients=gradients)
        self.theta_names = ["D", "log_floor_par", "log_floor_perp"] + [f"log_s_{t.name}" for t in self.tracks]
        ng = len(self.gps.y)
        self._rows = np.cumsum([0, ng] + [len(t.y) for t in self.tracks])
        self._qperp = [t.pE * self.n[0] + t.pN * self.n[1] for t in self.tracks]
        self._shapes = [(2 * ng, self.K)] + [(len(t.y), self.K) for t in self.tracks]
        lo, hi = self.bounds["D"]
        self._spl = DepthSpline(self._white_flat, lo, hi, dlog)

    def unpack(self, theta):
        th = np.asarray(theta, float)
        return th[0], None, np.exp(th[1:3]), np.exp(th[3:3 + self.T])

    def _columns_at(self, D):
        Ux, Uy, Uz = self.green(D)
        r = self._rows
        Vg = np.r_[Ux[r[0]:r[1]], Uy[r[0]:r[1]]]
        Vt = []
        for t, (tr, pre) in enumerate(zip(self.tracks, self.tr_pre)):
            a, b = r[t + 1], r[t + 2]
            Vt.append(pre["sp"][:, None] * Ux[a:b] + self._qperp[t][:, None] * Uy[a:b] + tr.pU[:, None] * Uz[a:b])
        return Vg, Vt

    def _white_flat(self, D):
        Vg, Vt = self._columns_at(D)
        W = [solve_triangular(pre["L"], v, lower=True) for v, pre in zip(Vt, self.tr_pre)]
        return np.concatenate([Vg.ravel()] + [w.ravel() for w in W])

    def _fault_columns(self, theta):
        return self._columns_at(float(np.asarray(theta, float)[0]))

    def _fault_columns_white(self, theta):
        flat = self._spl(float(np.asarray(theta, float)[0]))
        out, i = [], 0
        for shp in self._shapes:
            n = shp[0] * shp[1]; out.append(flat[i:i + n].reshape(shp)); i += n
        return out[0], out[1:]
