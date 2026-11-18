"""3B ileri model: homojen, izotrop elastik yarı uzayda dikdörtgen dislokasyonun YÜZEY (z = 0)
yer değiştirmesi / hızı. Okada (1985, BSSA 75(4), 1135-1154) kapalı-biçim yüzey çözümü.

Uzlaşımlar (testlerle kilitli: tests/test_okada.py)
---------------------------------------------------
* Birimler    : konum ve uzunluklar km; kayma mm/yıl -> çıktı mm/yıl (çıktı kayma birimindedir).
* strike      : fay doğrultusu azimutu, kuzeyden saat yönünde (derece). s = (sin a, cos a);
                n = s'nin 90° saat yönü tersine döndürülmüşü = SOL taraf (geometry.Profile ile aynı).
* dip         : sağ-el kuralı; fay doğrultuya bakan gözlemcinin SAĞINA (−n yönüne) eğimlidir,
                0 < dip <= 90.
* ref_e, ref_n: fayın ÜST kenarı orta noktasının yüzey izdüşümü (dip = 90 için iz orta noktası).
                Üst kenar ref'ten s boyunca ±length/2 uzanır; alt kenarın izdüşümü
                (bottom − top)/tan(dip) km sağdadır.
* top, bottom : üst / alt kenar derinliği (km, aşağı pozitif), 0 <= top < bottom. Yarı-sonsuz
                derin kayma için bottom çok büyük verilebilir (örn. 1e5).
* ss > 0      : SAĞ-YANAL (forward.py'deki V > 0 ile aynı: sol blok sağ bloğa göre +s yönünde).
                DİKKAT: Okada'nın U1'i SOL-YANAL pozitiftir; burada U1 = −ss.
* ds > 0      : ters / bindirme: tavan bloğu (doğrultunun sağı) tabana göre yukarı-eğim yönünde
                gider. Okada U2 = ds. (Düşey fayda "tavan" = sağ blok; sağ blok yukarı.)
* nu          : Poisson oranı (varsayılan 0.25, yani lambda = mu); mu/(lambda+mu) = 1 − 2 nu.
* Çıktı       : (ue, un, uu) = doğu, kuzey, yukarı; e ve n'nin yayın (broadcast) şeklinde.

Okada yerel çerçevesi: x = s, y = n (sol), z yukarı; fay −y yönüne (sağa) eğimlidir. Chinnery
toplamı f(xi, eta)|| dört köşe üzerinden alınır; q fay düzlemine dik uzaklıktır.

Sayısal ayrıntılar
------------------
* İptalsiz: eta < 0 için R+eta = (xi²+q²)/(R−eta); xi < 0 için R+xi = (eta²+q²)/(R−xi).
* ss, ds, nu skalerdir; gözlem noktaları vektörleştirilir (5000 nokta ~2 ms).
* I4 log1p ile, I5 ise ±(pi/2)·sgn(xi) sabiti atılarak (Chinnery toplamında tam sadeleşir)
  arctan2 ile yazılır; böylece dip -> 90'a yaklaşırken 1/cos(dip) iptalleri büyümez.
  |cos dip| < 1e-8 olduğunda Okada'nın cos(dip) = 0 dalı kullanılır.
* q = 0 (gözlem noktası fay düzleminin yüzey uzantısında): atan(xi·eta/(q·R)) := 0. Gömülü fayda
  bu tam doğrudur (alan süreklidir); yüzeyi kesen fayın tam iz üzerinde doğrultu atımlı çözüm iki
  tarafın ORTALAMASINI verir. Yüzeyi kesen fayda tam iz üzerinde eğim atımlı (ds) değerler ve
  yüzeydeki fay köşeleri (R = 0) tekildir; köşelerde NaN dönebilir.
"""
import numpy as np

_COS_EPS = 1e-8


def okada_rect(e, n, ref_e, ref_n, strike_deg, length, top, bottom, dip_deg=90.0, ss=0.0, ds=0.0, nu=0.25):
    """Dikdörtgen fayın yüzey yer değiştirmesi/hızı (doğu, kuzey, yukarı); bkz. modül belgesi.

    e, n : gözlem noktaları (km, yerel doğu/kuzey; diziler yayınlanır).
    """
    e, n = np.broadcast_arrays(np.asarray(e, float), np.asarray(n, float))
    length, top, bottom, dip_deg, nu = float(length), float(top), float(bottom), float(dip_deg), float(nu)
    if not length > 0:
        raise ValueError("length > 0 olmalı")
    if not 0.0 <= top < bottom:
        raise ValueError("0 <= top < bottom olmalı")
    if not 0.0 < dip_deg <= 90.0:
        raise ValueError("0 < dip_deg <= 90 olmalı (sağ-el kuralı)")
    if not -1.0 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 olmalı")

    a = np.deg2rad(float(strike_deg))
    s_e, s_n = np.sin(a), np.cos(a)
    n_e, n_n = -np.cos(a), np.sin(a)
    d = np.deg2rad(dip_deg)
    sd, cd = np.sin(d), np.cos(d)
    if abs(cd) < _COS_EPS:
        sd, cd = 1.0, 0.0

    de, dn = e - float(ref_e), n - float(ref_n)
    xs = de * s_e + dn * s_n            # doğrultu boyunca, üst kenar ortasından
    yn = de * n_e + dn * n_n            # sol-normal, üst kenar izdüşümünden
    W = (bottom - top) / sd             # eğim boyunca genişlik

    # Okada: p = y cos + d sin (alt kenara göre). Üst kenara göre yazınca iptal yok:
    q = yn * sd - top * cd
    eta_top = yn * cd + top * sd        # = p − W
    eta_bot = eta_top + W               # = p
    corners = ((xs + 0.5 * length, eta_bot, yn + W * cd, bottom, +1.0),
               (xs + 0.5 * length, eta_top, yn, top, -1.0),
               (xs - 0.5 * length, eta_bot, yn + W * cd, bottom, -1.0),
               (xs - 0.5 * length, eta_top, yn, top, +1.0))

    U1, U2 = -float(ss), float(ds)      # Okada: U1 > 0 sol-yanal; burada ss > 0 sağ-yanal
    ux = np.zeros_like(xs); uy = np.zeros_like(xs); uz = np.zeros_like(xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        for xi, eta, ytil, dtil, sign in corners:
            fx, fy, fz = _corner(xi, eta, q, ytil, dtil, sd, cd, 1.0 - 2.0 * nu, U1, U2)
            ux += sign * fx; uy += sign * fy; uz += sign * fz
    k = -1.0 / (2.0 * np.pi)
    ux, uy, uz = k * ux, k * uy, k * uz
    return ux * s_e + uy * n_e, ux * s_n + uy * n_n, uz


def _corner(xi, eta, q, ytil, dtil, sd, cd, a, U1, U2):
    """Okada (1985) yüzey ifadelerinin köşe değeri (köşeli parantez içi), U1 ve U2 ağırlıklı.
    ytil = eta cos + q sin, dtil = eta sin − q cos (köşenin derinliği: top ya da bottom)."""
    R = np.sqrt(xi * xi + eta * eta + q * q)
    X = np.sqrt(xi * xi + q * q)
    Rpe = np.where(eta >= 0, R + eta, (xi * xi + q * q) / (R - eta))     # R + eta
    Rpx = np.where(xi >= 0, R + xi, (eta * eta + q * q) / (R - xi))      # R + xi
    Rpd = R + dtil                                                       # dtil >= 0
    lnRpe = np.log(Rpe)
    iRpe = np.where(Rpe > 0, 1.0 / Rpe, 0.0)
    iRpx = np.where(Rpx > 0, 1.0 / Rpx, 0.0)
    th = np.where(q == 0, 0.0, np.arctan(xi * eta / (q * R)))

    if cd == 0.0:
        I1 = -0.5 * a * xi * q / Rpd**2
        I3 = 0.5 * a * (eta / Rpd + ytil * q / Rpd**2 - lnRpe)
        I4 = -a * q / Rpd
        I5 = -a * xi * sd / Rpd
    else:
        # I4 = a/cos [ln(R+dtil) − sin ln(R+eta)];  (R+dtil)/(R+eta) − 1 = −cos(eta cos/(1+sin) + q)/(R+eta)
        I4 = a * (np.log1p(-cd * (eta * cd / (1.0 + sd) + q) * iRpe) / cd + cd / (1.0 + sd) * lnRpe)
        # I5 = 2a/cos atan(N / (xi (R+X) cos)) − (pi a/cos) sgn(xi)   [sabit, toplamda sadeleşir]
        N = eta * (X + q * cd) + X * (R + X) * sd
        I5 = np.where(xi == 0, 0.0, -2.0 * a / cd * np.arctan2(xi * (R + X) * cd, N))
        I3 = a * (ytil / (cd * Rpd) - lnRpe) + sd / cd * I4
        I1 = -a * xi / (cd * Rpd) - sd / cd * I5
    I2 = -a * lnRpe - I3

    iR = 1.0 / R
    fx = U1 * (xi * q * iR * iRpe + th + I1 * sd) + U2 * (q * iR - I3 * sd * cd)
    fy = U1 * (ytil * q * iR * iRpe + q * cd * iRpe + I2 * sd) + U2 * (ytil * q * iR * iRpx + cd * th - I1 * sd * cd)
    fz = U1 * (dtil * q * iR * iRpe + q * sd * iRpe + I4 * sd) + U2 * (dtil * q * iR * iRpx + sd * th - I5 * sd * cd)
    return fx, fy, fz
