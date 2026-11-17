"""Profil geometrisi uzlaşımları.

Uzlaşım: strike = fay doğrultusunun azimutu (kuzeyden saat yönünde, derece, herhangi bir değer).
s = doğrultu birim vektörü (doğu, kuzey); n = s'nin 90° saat yönü TERSİNE döndürülmüşü, yani
doğrultuya bakan gözlemcinin SOL tarafı. x = s yönünde, y = n yönünde mesafe (km).
"""
import numpy as np
import pytest

from profil2d.geometry import Profile


def test_left_normal_is_north_for_ene_strike():
    p = Profile(lon0=27.6, lat0=39.6, strike=77.0)
    assert p.n[1] > 0.9          # kuzey bileşeni baskın ve pozitif
    x, y = p.to_profile(np.array([27.6]), np.array([39.7]))   # ~11 km kuzey
    assert y[0] == pytest.approx(11.1 * np.sin(np.deg2rad(77.0)), rel=0.02)


def test_unit_vectors_orthonormal_any_strike():
    for st in [-200, -103, 0, 77, 180, 257, 359.9, 400]:
        p = Profile(27.6, 39.6, st)
        assert np.dot(p.s, p.s) == pytest.approx(1)
        assert np.dot(p.n, p.n) == pytest.approx(1)
        assert np.dot(p.s, p.n) == pytest.approx(0, abs=1e-12)
        assert p.s[0] * p.n[1] - p.s[1] * p.n[0] == pytest.approx(1)   # n, s'nin solunda


def test_projection_roundtrip_and_distance():
    p = Profile(27.6, 39.6, 77.0)
    lon = np.array([27.0, 27.6, 28.3]); lat = np.array([39.3, 39.6, 39.9])
    e, n = p.to_local(lon, lat)
    lon2, lat2 = p.from_local(e, n)
    assert np.allclose(lon, lon2, atol=1e-9) and np.allclose(lat, lat2, atol=1e-9)
    assert e[1] == pytest.approx(0, abs=1e-6) and n[1] == pytest.approx(0, abs=1e-6)


def test_vector_rotation_roundtrip():
    p = Profile(27.6, 39.6, 77.0)
    ve, vn = np.array([1.0, -3.0]), np.array([2.0, 0.5])
    vpar, vperp = p.enu_to_profile(ve, vn)
    ve2, vn2 = p.profile_to_enu(vpar, vperp)
    assert np.allclose(ve, ve2) and np.allclose(vn, vn2)
