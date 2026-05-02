from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from genet.model import Driver, GENET


def _make_time_index(n=6, start="2024-01-01 00:00:00"):
    return pd.date_range(start=start, periods=n, freq="1h", tz="UTC")


def test_driver_extract_current():
    idx = _make_time_index()
    df = pd.DataFrame({"sme": np.arange(len(idx), dtype=float)}, index=idx)

    out = Driver("sme", "current").extract(df)

    assert out.name == "sme - current"
    pd.testing.assert_series_equal(out, df["sme"].rename("sme - current"))


def test_driver_extract_mean():
    idx = _make_time_index(8)
    df = pd.DataFrame({"sme": np.arange(8, dtype=float)}, index=idx)
    d = Driver("sme", "mean", (-2.0, -1.0))

    out = d.extract(df)

    shifted = df["sme"].shift(freq=pd.to_timedelta(2.0, unit="h"))
    expected = shifted.rolling(pd.to_timedelta(1.0, unit="h")).mean()
    expected = expected.shift(freq=-pd.to_timedelta(1.0, unit="h"))
    expected.name = "sme - mean -2.0 -1.0"
    pd.testing.assert_series_equal(out, expected)


def test_driver_extract_max():
    idx = _make_time_index(8)
    df = pd.DataFrame({"hp30": [2.0, 5.0, 1.0, 7.0, 3.0, 0.0, 9.0, 4.0]}, index=idx)
    d = Driver("hp30", "max", (-2.0, -1.0))

    out = d.extract(df)

    shifted = df["hp30"].shift(freq=pd.to_timedelta(2.0, unit="h"))
    expected = shifted.rolling(pd.to_timedelta(1.0, unit="h")).max()
    expected = expected.shift(freq=-pd.to_timedelta(1.0, unit="h"))
    expected.name = "hp30 - max -2.0 -1.0"
    pd.testing.assert_series_equal(out, expected)


def test_driver_extract_fraction_negative():
    idx = _make_time_index(8)
    vals = np.array([1.0, -1.0, 2.0, -3.0, -5.0, 4.0, 0.0, -2.0])
    df = pd.DataFrame({"by_gsm": vals}, index=idx)
    d = Driver("by_gsm", "fraction_negative", (-2.0, -1.0))

    out = d.extract(df)

    shifted = df["by_gsm"].shift(freq=pd.to_timedelta(2.0, unit="h"))
    expected = shifted.rolling(pd.to_timedelta(1.0, unit="h")).apply(
        lambda x: (x < 0).mean()
    )
    expected = expected.shift(freq=-pd.to_timedelta(1.0, unit="h"))
    expected.name = "by_gsm - fraction_negative -2.0 -1.0"
    pd.testing.assert_series_equal(out, expected)


def test_driver_extract_unsupported_function():
    idx = _make_time_index()
    df = pd.DataFrame({"sme": np.arange(len(idx), dtype=float)}, index=idx)
    d = Driver("sme", "current")
    object.__setattr__(d, "func", "bad")

    with pytest.raises(ValueError, match="Unsupported aggregate function"):
        d.extract(df)


def test_gelu():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)

    gelu = GENET._gelu(x)

    assert gelu.shape == x.shape
    assert np.all(np.diff(gelu) >= 0)


def test_relu():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)

    relu = GENET._relu(x)

    np.testing.assert_array_equal(relu, np.array([0.0, 0.0, 2.0]))


def test_to_real_scale():
    g = GENET.__new__(GENET)

    scaled = g._to_real_scale(np.array([0.0, 1.0], dtype=np.float64))

    np.testing.assert_allclose(scaled, np.array([0.0, 9.0]))


def test_to_utc_datetime():
    naive = datetime(2024, 1, 1, 12, 0, 0)
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    out_naive = GENET._to_utc_datetime(naive)
    out_aware = GENET._to_utc_datetime(aware)

    assert out_naive.tzinfo == timezone.utc
    assert out_aware.tzinfo == timezone.utc


def test_interpolate_missing_all_nan():
    s_all_nan = pd.Series([np.nan, np.nan, np.nan])

    out_nan = GENET._interpolate_missing(s_all_nan)

    np.testing.assert_array_equal(out_nan, np.zeros(3))


def test_interpolate_missing_single_point():
    s_single = pd.Series([np.nan, 5.0, np.nan, np.nan])

    out_single = GENET._interpolate_missing(s_single)

    np.testing.assert_array_equal(out_single, np.array([5.0, 5.0, 5.0, 5.0]))


def test_interpolate_missing_multiple_points():
    s = pd.Series([1.0, np.nan, np.nan, 4.0])

    out = GENET._interpolate_missing(s)

    assert out.shape == (4,)
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[-1], 4.0)


def test_as_list_and_is_single_coord():
    assert GENET._as_list(3) == [3]
    assert GENET._as_list(np.array([1, 2])).__class__ is list

    assert GENET._is_single_coord((1, 2, 3))
    assert GENET._is_single_coord(np.array([1.0, 2.0, 3.0]))
    assert not GENET._is_single_coord([1, 2])
    assert not GENET._is_single_coord(np.array([[1, 2, 3]]))


def _new_genet(monkeypatch, tmp_path):
    monkeypatch.setattr("genet.model.user_cache_dir", lambda _name: str(tmp_path))
    return GENET("demo")


def test_normalize_coords(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    assert g._normalize_coords((1, 2, 3)) == [(1, 2, 3)]
    assert g._normalize_coords(np.array([1, 2, 3])) == [(1, 2, 3)]
    assert g._normalize_coords(np.array([[1, 2, 3], [4, 5, 6]])) == [
        (1, 2, 3),
        (4, 5, 6),
    ]

    with pytest.raises(ValueError, match=r"shape \(3,\) or \(N, 3\)"):
        g._normalize_coords(np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="list must be a single"):
        g._normalize_coords([[1, 2, 3], [1, 2]])


def test_broadcast(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    assert g._broadcast([1], 3, "x") == [1, 1, 1]
    assert g._broadcast([1, 2], 2, "x") == [1, 2]
    with pytest.raises(ValueError, match="length 1 or 3"):
        g._broadcast([1, 2], 3, "x")


def test_validators(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    g._validate_coords_range(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="coords_gsm out of model coverage"):
        g._validate_coords_range(np.array([[21.0, 0.0, 0.0]], dtype=np.float32))

    g._validate_energy_range(np.array([0.1, 80.0], dtype=np.float32))
    with pytest.raises(ValueError, match="energy out of model range"):
        g._validate_energy_range(np.array([0.09], dtype=np.float32))

    g._validate_pitch_angle_range(np.array([10.0, 170.0], dtype=np.float32))
    with pytest.raises(ValueError, match="pitch_angle out of model range"):
        g._validate_pitch_angle_range(np.array([9.0], dtype=np.float32))


def test_validate_data(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="not available"):
        g._validate_data(pd.DataFrame(), "omni", start, end)

    with pytest.raises(ValueError, match="not available"):
        g._validate_data(pd.Series([np.nan, np.nan]), "sme", start, end)


def test_build_drivers(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)
    idx = pd.date_range("2024-01-01", periods=2000, freq="1min", tz="UTC")
    data = pd.DataFrame(
        {
            "sme": np.linspace(1, 2, len(idx)),
            "hp30": np.linspace(2, 3, len(idx)),
            "speed": np.linspace(300, 450, len(idx)),
            "proton_density": np.linspace(4, 8, len(idx)),
            "bavg": np.linspace(3, 7, len(idx)),
            "bx_gsm": np.linspace(-2, 2, len(idx)),
            "by_gsm": np.linspace(-3, 3, len(idx)),
            "bz_gsm": np.linspace(-4, 4, len(idx)),
        },
        index=idx,
    )
    data.index.name = "time"
    times = pd.DatetimeIndex([idx[100], idx[500], idx[1000]])

    out = g._build_drivers(data, times)

    assert out.shape == (3, 36)
    assert out.dtype == np.float32


def test_static_and_dynamic_branches(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)
    n = 4

    x_static = np.ones((n, 6), dtype=np.float32)
    kernels = [
        np.ones((6, 5), dtype=np.float32),
        np.ones((5, 4), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
        np.ones((3, 9), dtype=np.float32),
        np.ones((51, 7), dtype=np.float32),
        np.ones((7, 6), dtype=np.float32),
        np.ones((6, 5), dtype=np.float32),
        np.ones((5, 9), dtype=np.float32),
    ]
    biases = [
        np.zeros(5, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(9, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.zeros(9, dtype=np.float32),
    ]

    y_static = g._static_branch(x_static, kernels, biases)
    assert y_static.shape == (n, 9)

    x_full = np.ones((n, 42), dtype=np.float32)
    y_dynamic = g._dynamic_branch(x_full, y_static, kernels, biases)
    assert y_dynamic.shape == (n, 9)


def test_get_model_weights_cache(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    k1, b1 = g._get_model_weights("50")
    k2, b2 = g._get_model_weights("50")

    assert k1 is k2
    assert b1 is b2


def test_predict_invalid(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="percentile must be one of"):
        g.predict(datetime(2024, 1, 1), (1, 0, 0), 1.0, 90.0, percentile="10")

    monkeypatch.setattr(g, "_get_model_weights", lambda _p: ([], []))
    with pytest.raises(ValueError, match="pitch_angle string must be 'omnidirectional'"):
        g.predict(datetime(2024, 1, 1), (1, 0, 0), 1.0, "bad", percentile="50")


def test_predict(monkeypatch, tmp_path):
    g = _new_genet(monkeypatch, tmp_path)

    n = 2
    x_full = np.zeros((n, 42), dtype=np.float32)

    def fake_get_weights(_p):
        kernels = [
            np.ones((6, 5), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
            np.ones((3, 9), dtype=np.float32),
            np.ones((51, 7), dtype=np.float32),
            np.ones((7, 6), dtype=np.float32),
            np.ones((6, 5), dtype=np.float32),
            np.ones((5, 9), dtype=np.float32),
        ]
        biases = [
            np.zeros(5, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(9, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            np.zeros(9, dtype=np.float32),
        ]
        return kernels, biases

    monkeypatch.setattr(g, "_get_model_weights", fake_get_weights)
    monkeypatch.setattr(g, "_load_data", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(g, "_build_drivers", lambda data, times: x_full[:, 6:])
    monkeypatch.setattr("genet.model.geopack.recalc", lambda _t: 0.0)

    monkeypatch.setattr(g, "_static_branch", lambda x, k, b: np.ones((n, 9), dtype=np.float32))
    monkeypatch.setattr(g, "_dynamic_branch", lambda x, y, k, b: np.ones((n, 9), dtype=np.float32))

    times = [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
    ]
    coords = [(2.0, 0.0, 0.0), (3.0, 1.0, 1.0)]
    energy = [1.0, 2.0]

    y_angle = g.predict(times, coords, energy, [30.0, 90.0], percentile="50")
    y_omni = g.predict(times, coords, energy, "omnidirectional", percentile="50")

    assert y_angle.shape == (2,)
    assert y_omni.shape == (2,)
    assert np.all(y_angle >= 0)
    assert np.all(y_omni >= 0)
