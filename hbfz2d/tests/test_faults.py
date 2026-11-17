"""Fay izi (GeoJSON) okuma ve profil üzerindeki konumlar."""
import json
import numpy as np
import pytest

from profil2d.geometry import Profile
from profil2d.faults import read_traces, principal_strike, fault_offsets


def _line(lon0, lat0, az_deg, length_km, npts=50):
    p = Profile(lon0, lat0, 0.0)
    t = np.linspace(-length_km / 2, length_km / 2, npts)
    a = np.deg2rad(az_deg)
    return np.c_[p.from_local(t * np.sin(a), t * np.cos(a))]


@pytest.fixture
def geojson(tmp_path):
    f1 = _line(27.6, 39.70, 77.0, 80.0)            # kuzeydeki fay
    f2 = _line(27.6, 39.60, 77.0 + 180.0, 60.0)    # güneydeki fay, ters sayısallaştırılmış
    feats = [dict(type="Feature", properties=dict(FAY_ADI="KUZEY"), geometry=dict(type="LineString", coordinates=[[*c, 0] for c in f1.tolist()])),
             dict(type="Feature", properties=dict(FAY_ADI="GUNEY"), geometry=dict(type="LineString", coordinates=[[*c, 0] for c in f2.tolist()]))]
    p = tmp_path / "f.geojson"; p.write_text(json.dumps(dict(type="FeatureCollection", features=feats)))
    return str(p)


def test_read_traces_groups_by_name(geojson):
    tr = read_traces(geojson, name_key="FAY_ADI")
    assert set(tr) == {"KUZEY", "GUNEY"} and len(tr["KUZEY"]) == 1 and tr["KUZEY"][0].shape[1] == 2


def test_principal_strike_ignores_digitizing_direction(geojson):
    tr = read_traces(geojson, name_key="FAY_ADI")
    assert principal_strike(tr) == pytest.approx(77.0, abs=0.5)     # 0–180 aralığına indirgenir


def test_fault_offsets_sign_and_size(geojson):
    tr = read_traces(geojson, name_key="FAY_ADI")
    p = Profile(27.6, 39.65, 77.0)
    off = fault_offsets(tr, p)
    sep = 11.1 * np.sin(np.deg2rad(77.0))           # 0.1° enlem farkının faya dik bileşeni (~10.8 km)
    assert off["KUZEY"] == pytest.approx(+sep / 2, abs=0.4)
    assert off["GUNEY"] == pytest.approx(-sep / 2, abs=0.4)
