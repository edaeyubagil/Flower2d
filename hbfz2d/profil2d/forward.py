"""İleri model: 2B düşey doğrultu atımlı faylar (vida dislokasyonu, Savage & Burford 1973).

Uzlaşımlar (testlerle kilitli: tests/test_forward.py, tests/test_flower2d_crosscheck.py)
---------------------------------------------------------------------------------------
* V > 0  : SAĞ-YANAL kayma hızı (mm/yıl). Sol-yanal için V < 0.  (Flower2d: ss = -V)
* D      : kilitlenme derinliği (km); fay D'nin altında V hızıyla kayar, üstü kilitli.
* y0     : fayın profil üzerindeki konumu (km, Profile.y ekseni).
* v_par(y) = sum_i (V_i / pi) * atan((y - y0_i) / D_i)      [s yönünde, mm/yıl]
* LOS    : yer -> uydu birim vektörüyle iç çarpım; POZİTİF = uyduya doğru (menzil azalması).
* heading: UÇUŞ yönü azimutu (kuzeyden saat yönünde); sağa bakan SAR varsayılır.
* incidence: dikeyden ölçülen geliş açısı (derece).

`lib` argümanı numpy (varsayılan) ya da pytensor.tensor olabilir; böylece aynı formül
PyMC içinde türevlenebilir biçimde kullanılır.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fault:
    y0: float
    D: float
    V: float


def screw(y, y0, D, V, lib=np):
    return V / np.pi * lib.arctan((y - y0) / D)


def fault_parallel(y, faults, lib=np):
    """Faya paralel hız (mm/yıl), tüm fayların toplamı."""
    out = 0.0
    for f in faults:
        out = out + screw(y, f.y0, f.D, f.V, lib=lib)
    return out


def enu_velocity(profile, lon, lat, faults):
    """Fayların ürettiği yatay hız (doğu, kuzey) mm/yıl; faya dik ve düşey bileşen sıfır."""
    _, y = profile.to_profile(lon, lat)
    vpar = fault_parallel(y, faults)
    return profile.profile_to_enu(vpar, np.zeros_like(vpar))


def los_unit_vector(incidence, heading):
    """Yer -> uydu birim vektörü (doğu, kuzey, yukarı). Dizi girdileri desteklenir (son eksen yerine ilk eksen)."""
    inc = np.deg2rad(np.asarray(incidence, float))
    az = np.deg2rad(np.asarray(heading, float) - 90.0)   # sağa bakan SAR: yer->uydu yatay azimutu = uçuş - 90°
    return np.array([np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc) * np.ones_like(az)])


def los_velocity(profile, lon, lat, faults, incidence, heading, v_up=0.0):
    ve, vn = enu_velocity(profile, lon, lat, faults)
    u = los_unit_vector(incidence, heading)
    return ve * u[0] + vn * u[1] + v_up * u[2]
