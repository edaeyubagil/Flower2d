"""İleri model uzlaşımları: V > 0 = SAĞ-YANAL; LOS > 0 = uyduya doğru."""
import numpy as np
import pytest

from profil2d.geometry import Profile
from profil2d.forward import Fault, fault_parallel, enu_velocity, los_unit_vector, los_velocity


def test_far_field_is_half_slip_rate():
    v = fault_parallel(np.array([-1e7, 1e7]), [Fault(y0=0, D=12, V=4)])
    assert v == pytest.approx([-2, 2], abs=1e-5)


def test_right_lateral_sign_east_west_fault():
    # D–B uzanımlı fay; sağ-yanal: kuzey blok güneye göre DOĞUYA gider
    p = Profile(27.6, 39.6, strike=90.0)
    lon = np.array([27.6, 27.6]); lat = np.array([39.2, 40.0])      # güney, kuzey
    ve, vn = enu_velocity(p, lon, lat, [Fault(y0=0, D=10, V=+5)])
    assert ve[1] - ve[0] > 4.0            # kuzey − güney doğu farkı ≈ +V (uzak alanda)
    assert np.allclose(vn, 0, atol=1e-9)


@pytest.mark.parametrize("strike_a, strike_b", [(77, 257), (90, -90), (0, 180)])
def test_reversed_strike_gives_same_enu(strike_a, strike_b):
    """Doğrultu ters yazılsa da aynı fiziksel (sağ-yanal) fay aynı E,N alanını vermeli.
    (Flower2d'de 292 girilince yp yönü dönüyordu; burada fay konumu coğrafi olarak verilir.)"""
    lon = np.linspace(27.0, 28.2, 9); lat = np.linspace(39.2, 40.1, 9)
    out = []
    for st in (strike_a, strike_b):
        p = Profile(27.6, 39.6, st)
        y_fault = p.to_profile(np.array([27.6]), np.array([39.65]))[1][0]
        out.append(enu_velocity(p, lon, lat, [Fault(y0=y_fault, D=12, V=3)]))
    assert np.allclose(out[0][0], out[1][0], atol=1e-9) and np.allclose(out[0][1], out[1][1], atol=1e-9)


def test_two_faults_superpose():
    y = np.linspace(-50, 50, 11)
    f1, f2 = Fault(y0=-3.6, D=10, V=1.5), Fault(y0=3.1, D=15, V=2.0)
    assert np.allclose(fault_parallel(y, [f1, f2]), fault_parallel(y, [f1]) + fault_parallel(y, [f2]))


def flower2d_proj(look, head):
    """Flower2d networkopti.py:213-217 ile aynı formül (head = BAKIŞ azimutu)."""
    phi = np.deg2rad(-90. - head); theta = np.deg2rad(90. - look)
    return np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])


@pytest.mark.parametrize("inc, heading", [(20.0, 194.0), (23.0, 347.0), (39.0, 190.5), (34.0, -10.0)])
def test_los_vector_matches_flower2d_with_heading_plus_90(inc, heading):
    """Bizim heading = UÇUŞ yönü; Flower2d'nin 'heading'i = bakış azimutu = uçuş + 90°."""
    assert np.allclose(los_unit_vector(inc, heading), flower2d_proj(inc, heading + 90.0), atol=1e-12)


def test_los_positive_toward_satellite():
    p = los_unit_vector(30.0, 194.0)           # alçalan
    assert p[2] > 0                              # yükselme -> pozitif LOS
    assert p[0] > 0                              # alçalan: uydu hedefin doğusunda
    assert los_unit_vector(30.0, 347.0)[0] < 0  # yükselen: uydu batıda


def test_los_velocity_is_enu_dot_unit_vector():
    p = Profile(27.6, 39.6, 77.0)
    lon = np.array([27.3, 27.9]); lat = np.array([39.5, 39.8])
    faults = [Fault(y0=0, D=12, V=3)]
    ve, vn = enu_velocity(p, lon, lat, faults)
    u = los_unit_vector(np.array([35.0, 38.0]), np.array([194.0, 194.0]))
    los = los_velocity(p, lon, lat, faults, incidence=np.array([35.0, 38.0]), heading=np.array([194.0, 194.0]))
    assert np.allclose(los, ve * u[0] + vn * u[1])
