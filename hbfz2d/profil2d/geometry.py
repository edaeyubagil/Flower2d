"""Profil geometrisi.

Uzlaşımlar (testlerle kilitli: tests/test_geometry.py)
------------------------------------------------------
* strike  : fay doğrultusunun azimutu, kuzeyden saat yönünde (derece). Her değer kabul edilir.
* s       : doğrultu birim vektörü (doğu, kuzey) = (sin a, cos a).
* n       : s'nin 90° saat yönü TERSİNE döndürülmüşü = doğrultuya bakan gözlemcinin SOL tarafı.
* x, y    : s ve n yönündeki mesafeler (km); merkez (lon0, lat0).
* İzdüşüm : merkezde azimutal eşit uzaklıklı (aeqd, WGS84); 150 km içinde ölçek hatası < %0.03.
"""
import numpy as np
import pyproj


class Profile:
    def __init__(self, lon0, lat0, strike):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.strike = float(strike) % 360.0
        a = np.deg2rad(self.strike)
        self.s = np.array([np.sin(a), np.cos(a)])
        self.n = np.array([-np.cos(a), np.sin(a)])
        self._proj = pyproj.Proj(proj="aeqd", lat_0=self.lat0, lon_0=self.lon0, datum="WGS84")

    def to_local(self, lon, lat):
        """(lon, lat) -> (doğu, kuzey) km."""
        e, n = self._proj(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(e) / 1e3, np.asarray(n) / 1e3

    def from_local(self, e, n):
        lon, lat = self._proj(np.asarray(e, float) * 1e3, np.asarray(n, float) * 1e3, inverse=True)
        return np.asarray(lon), np.asarray(lat)

    def to_profile(self, lon, lat):
        """(lon, lat) -> (x, y) km; x doğrultu boyunca, y sol-normal yönünde."""
        e, n = self.to_local(lon, lat)
        return e * self.s[0] + n * self.s[1], e * self.n[0] + n * self.n[1]

    def enu_to_profile(self, ve, vn):
        """Yatay hız (doğu, kuzey) -> (faya paralel [s yönü], faya dik [n yönü])."""
        ve, vn = np.asarray(ve, float), np.asarray(vn, float)
        return ve * self.s[0] + vn * self.s[1], ve * self.n[0] + vn * self.n[1]

    def profile_to_enu(self, vpar, vperp):
        vpar, vperp = np.asarray(vpar, float), np.asarray(vperp, float)
        return vpar * self.s[0] + vperp * self.n[0], vpar * self.s[1] + vperp * self.n[1]
