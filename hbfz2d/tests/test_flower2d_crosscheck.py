"""Bağımsız uygulama kontrolü: aynı düşey fay için Flower2d'nin kendi kodu ile aynı E,N alanı.
Flower2d uzlaşımı: ss > 0 sol-yanal  ->  bizim V = -ss."""
import os, sys
import numpy as np
import pytest

from profil2d.geometry import Profile
from profil2d.forward import Fault, enu_velocity

FLOWER2D_SRC = os.environ.get("FLOWER2D_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "src", "flower2d"))


@pytest.fixture(scope="module")
def modelopti():
    if not os.path.isdir(FLOWER2D_SRC):
        pytest.skip("Flower2d kaynak klasörü yok")
    sys.path.insert(0, os.path.abspath(FLOWER2D_SRC))
    import modelopti as m
    return m


@pytest.mark.parametrize("V, D", [(3.0, 12.0), (-2.0, 7.5)])
def test_same_enu_as_flower2d_vertical_fault(modelopti, V, D):
    strike = 77.0
    p = Profile(27.6, 39.6, strike)
    lon = np.linspace(27.0, 28.2, 15); lat = np.linspace(39.3, 39.95, 15)
    ve, vn = enu_velocity(p, lon, lat, [Fault(y0=0.0, D=D, V=V)])

    # Flower2d: düşey (dip=90), yarı-sonsuz ana fay, kısalma 0
    mf = modelopti.mainfault(name="F", ss=-V, sigmass=0., short=0., sigmashort=0., w=D, sigmaw=0., dip=90.)
    inv = modelopti.inversion(structures=[mf], strike=strike, profile=modelopti.profile(name="P", x=0., y=0., l=400., w=400.))
    seg = mf.segments[0]; mf.conservation()
    e, n = p.to_local(lon, lat)
    yp_f = e * inv.n[0] + n * inv.n[1]             # Flower2d profil koordinatı (networkopti.py:170)
    u = seg.displacement(yp_f)                      # (par, perp, düşey) Flower2d s,n tabanında
    ve_f = u[:, 0] * inv.s[0] + u[:, 1] * inv.n[0]
    vn_f = u[:, 0] * inv.s[1] + u[:, 1] * inv.n[1]
    assert np.allclose(ve, ve_f, atol=1e-10) and np.allclose(vn, vn_f, atol=1e-10)
