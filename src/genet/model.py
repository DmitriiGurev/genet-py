from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from platformdirs import user_cache_dir
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

from geopack import geopack

DriverType = Literal["sme", "hp30", "speed", "proton_density", "bavg", "bx_gsm", "by_gsm", "bz_gsm"]
AggregateFunction = Literal["current", "mean", "max", "fraction_negative"]


@dataclass(frozen=True)
class Driver:
    type: DriverType
    func: AggregateFunction = "current"
    window_hours: tuple[float, float] = (0.0, 0.0)

    def extract(self, data: pd.DataFrame) -> pd.Series:
        series = data[self.type]

        if self.func == "current":
            result = series.copy()
            result.name = f"{self.type} - current"
            return result

        window_left, window_right = self.window_hours
        window_width = pd.to_timedelta(window_right - window_left, unit="h")
        result = series.shift(freq=pd.to_timedelta(-window_left, unit="h"))
        rolled = result.rolling(window=window_width)

        if self.func == "mean":
            result = rolled.mean()
        elif self.func == "max":
            result = rolled.max()
        elif self.func == "fraction_negative":
            result = rolled.apply(lambda x: (x < 0).mean())
        else:
            raise ValueError(f"Unsupported aggregate function: {self.func}")

        result = result.shift(freq=-window_width)
        result.name = f"{self.type} - {self.func} {window_left} {window_right}"
        return result


_DRIVER_LIST = [
    Driver("sme"),
    Driver("hp30"),
    Driver("sme", "mean", (-0.25, 0.0)),
    Driver("by_gsm", "fraction_negative", (-0.75, -0.5)),
    Driver("hp30", "mean", (-1.0, -0.75)),
    Driver("sme", "mean", (-1.5, -1.0)),
    Driver("proton_density"),
    Driver("proton_density", "max", (-0.25, 0.0)),
    Driver("proton_density", "mean", (-4.0, -3.0)),
    Driver("speed", "mean", (-6.0, -4.0)),
    Driver("proton_density", "mean", (-0.25, 0.0)),
    Driver("sme", "mean", (-2.0, -1.5)),
    Driver("speed"),
    Driver("bz_gsm", "fraction_negative", (-3.0, -2.0)),
    Driver("bz_gsm", "fraction_negative", (-4.0, -3.0)),
    Driver("sme", "max", (-1.5, -1.0)),
    Driver("bz_gsm", "fraction_negative", (-6.0, -4.0)),
    Driver("by_gsm", "fraction_negative", (-3.0, -2.0)),
    Driver("bavg", "mean", (-24.0, -16.0)),
    Driver("sme", "mean", (-0.5, -0.25)),
    Driver("hp30", "max", (-24.0, -16.0)),
    Driver("speed", "mean", (-0.25, 0.0)),
    Driver("proton_density", "max", (-0.5, -0.25)),
    Driver("proton_density", "mean", (-0.5, -0.25)),
    Driver("bz_gsm", "fraction_negative", (-0.5, -0.25)),
    Driver("speed", "mean", (-24.0, -16.0)),
    Driver("proton_density", "max", (-0.75, -0.5)),
    Driver("speed", "mean", (-0.5, -0.25)),
    Driver("bx_gsm", "fraction_negative", (-0.25, 0.0)),
    Driver("bavg", "max", (-6.0, -4.0)),
    Driver("bz_gsm", "fraction_negative", (-12.0, -8.0)),
    Driver("bavg", "max", (-0.25, 0.0)),
    Driver("sme", "mean", (-3.0, -2.0)),
    Driver("bz_gsm", "fraction_negative", (-1.5, -1.0)),
    Driver("bz_gsm", "fraction_negative", (-0.75, -0.5)),
    Driver("bz_gsm", "fraction_negative", (-2.0, -1.5)),
]


class GENET:
    def __init__(self, supermag_username: str) -> None:
        self.supermag_username = supermag_username
        self.cache_dir = Path(user_cache_dir("genet-py"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(__file__).resolve().parent / "models"

        scaler_npz = np.load(model_dir / "scaler.npz")
        self.scale = scaler_npz["scale"]
        self.min_ = scaler_npz["min"]

        weight_npz = np.load(model_dir / "weights.npz")
        self.kernels = [weight_npz[k] for k in sorted(k for k in weight_npz if k.startswith("kernel_"))]
        self.biases = [weight_npz[k] for k in sorted(k for k in weight_npz if k.startswith("bias_"))]

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + np.tanh(tanh_arg))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def _to_utc_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _interpolate_missing(series: pd.Series) -> np.ndarray:
        mask = series.notna()
        inds = np.flatnonzero(mask.to_numpy())
        if len(inds) == 0:
            return np.zeros(len(series), dtype=np.float64)
        if len(inds) == 1:
            return np.full(len(series), float(series.iloc[inds[0]]), dtype=np.float64)
        spline = PchipInterpolator(inds, series.iloc[inds].to_numpy())
        return spline(np.arange(len(series))).ravel()

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            return list(value)
        return [value]

    @staticmethod
    def _is_single_coord(value) -> bool:
        if isinstance(value, np.ndarray):
            return value.ndim == 1 and value.shape[0] == 3
        if isinstance(value, (tuple, list)) and len(value) == 3:
            return all(isinstance(v, (int, float, np.integer, np.floating)) for v in value)
        return False

    def _normalize_coords(self, coords_gsm) -> list:
        if self._is_single_coord(coords_gsm):
            return [tuple(coords_gsm)]

        if isinstance(coords_gsm, np.ndarray):
            if coords_gsm.ndim == 2 and coords_gsm.shape[1] == 3:
                return [tuple(row) for row in coords_gsm.tolist()]
            raise ValueError("coords_gsm ndarray must be shape (3,) or (N, 3)")

        if isinstance(coords_gsm, list):
            if all(self._is_single_coord(c) for c in coords_gsm):
                return [tuple(c) for c in coords_gsm]
            raise ValueError("coords_gsm list must be a single 3-element coordinate or contain only 3-element coordinates")

        raise ValueError("coords_gsm must be a 3-element coordinate or a list/array of such coordinates")

    @staticmethod
    def _broadcast(values: list, n: int, name: str) -> list:
        if len(values) == 1:
            return values * n
        if len(values) == n:
            return values
        raise ValueError(f"'{name}' must have length 1 or {n}, got {len(values)}")

    def _load_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        start = self._to_utc_datetime(start)
        end = self._to_utc_datetime(end)

        from swvo.io.solar_wind import SWOMNI
        from swvo.io.sme import SMESuperMAG
        from swvo.io.hp import Hp30GFZ

        omni = SWOMNI(self.cache_dir).read(start, end, download=True)
        omni.index.name = "time"
        if "file_name" in omni.columns:
            omni = omni.drop(columns="file_name")
        omni = omni.astype(np.float64)
        for col in omni.columns:
            omni.loc[:, col] = self._interpolate_missing(omni[col])

        sme = SMESuperMAG(self.supermag_username, self.cache_dir).read(start, end, download=True)
        sme.index.name = "time"
        if "file_name" in sme.columns:
            sme = sme.drop(columns="file_name")
        sme_col = sme.columns[0]
        sme = sme.rename(columns={sme_col: "sme"})
        sme = sme.astype(np.float64)
        sme.loc[:, "sme"] = self._interpolate_missing(sme["sme"])

        hp30 = Hp30GFZ(self.cache_dir).read(start, end, download=True)
        hp30.index.name = "time"
        if "file_name" in hp30.columns:
            hp30 = hp30.drop(columns="file_name")
        hp30_col = hp30.columns[0]
        hp30 = hp30.rename(columns={hp30_col: "hp30"}).resample("1min").mean()
        hp30 = hp30.astype(np.float64)
        hp30.loc[:, "hp30"] = self._interpolate_missing(hp30["hp30"])

        return omni.merge(sme, on="time").merge(hp30, on="time").sort_index()

    def _build_drivers(self, data: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
        out = None
        for driver in _DRIVER_LIST:
            s = driver.extract(data)
            df = s.reset_index().set_index("time")
            out = df if out is None else out.merge(df, on="time", how="outer")

        out = out.sort_index().interpolate(method="time").ffill().bfill()
        out = out.reindex(times).interpolate(method="time").ffill().bfill()
        return out.to_numpy(dtype=np.float32)

    def _static_branch(self, x: np.ndarray) -> np.ndarray:
        w, b = self.kernels, self.biases
        hl0 = self._gelu(x @ w[0] + b[0])
        hl1 = self._gelu(hl0 @ w[1] + b[1])
        hl2 = self._gelu(hl1 @ w[2] + b[2])
        return hl2 @ w[3] + b[3]

    def _dynamic_branch(self, x: np.ndarray, y_static: np.ndarray) -> np.ndarray:
        w, b = self.kernels, self.biases
        x_full = np.concatenate([x, y_static], axis=1)
        hl0 = self._gelu(x_full @ w[4] + b[4])
        hl1 = self._gelu(hl0 @ w[5] + b[5])
        hl2 = self._gelu(hl1 @ w[6] + b[6])
        return hl2 @ w[7] + b[7]

    def _to_real_scale(self, x: np.ndarray) -> np.ndarray:
        return 10**x - 1

    def predict(self, time, coords_gsm, energy, pitch_angle):
        time_values = [self._to_utc_datetime(t) for t in self._as_list(time)]
        coords_values = self._normalize_coords(coords_gsm)
        energy_values = self._as_list(energy)

        n = max(
            len(time_values),
            len(coords_values),
            len(energy_values),
            1 if isinstance(pitch_angle, str) else len(self._as_list(pitch_angle)),
        )

        times = self._broadcast(time_values, n, "time")
        coords = self._broadcast(coords_values, n, "coords_gsm")
        energies = self._broadcast(energy_values, n, "energy")

        if isinstance(pitch_angle, str):
            if pitch_angle != "omnidirectional":
                raise ValueError("pitch_angle string must be 'omnidirectional'")
            pitch_mode = "omnidirectional"
            pitch_values = None
        else:
            pitch_mode = "angle"
            pitch_values = np.asarray(self._broadcast(self._as_list(pitch_angle), n, "pitch_angle"), dtype=np.float32)

        times_index = pd.DatetimeIndex(times).sort_values()
        start = times_index.min() - timedelta(hours=24)
        end = times_index.max()

        data = self._load_data(start=start, end=end)
        drivers = self._build_drivers(data, pd.DatetimeIndex(times))

        coords_arr = np.asarray(coords, dtype=np.float32)
        energy_arr = np.asarray(energies, dtype=np.float32)

        x_gsm = coords_arr[:, 0]
        y_gsm = coords_arr[:, 1]
        z_gsm = coords_arr[:, 2]

        r = np.sqrt(x_gsm**2 + y_gsm**2 + z_gsm**2)
        lat = np.arcsin(z_gsm / r)
        mlt = (12.0 + np.degrees(np.arctan2(y_gsm, x_gsm)) / 15.0) % 24.0

        unix0 = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dipole_tilt = np.asarray(
            [geopack.recalc(pd.to_timedelta(t.to_pydatetime() - unix0).total_seconds()) for t in pd.DatetimeIndex(times)],
            dtype=np.float32,
        )

        x = np.empty((n, 42), dtype=np.float32)
        x[:, 0] = np.log10(energy_arr)
        x[:, 1] = r * 6378.0
        x[:, 2] = np.sin(lat)
        x[:, 3] = np.sin(mlt * np.pi / 12.0) * np.cos(lat)
        x[:, 4] = np.cos(mlt * np.pi / 12.0) * np.cos(lat)
        x[:, 5] = dipole_tilt
        x[:, 6:] = drivers

        x = x * self.scale + self.min_

        static_inds = [1, 2, 3, 4, 0, 5]
        y_static = self._static_branch(x[:, static_inds])
        y_dynamic = self._dynamic_branch(x, y_static)
        y = self._relu(y_static + y_dynamic)

        if pitch_mode == "omnidirectional":
            y_omnidirectional = np.sum([y[:, i] * np.sin(np.radians(10 + i * 20)) for i in range(9)], axis=0) / np.sum(np.sin(np.radians(np.arange(10, 180, 20))))
            return self._to_real_scale(y_omnidirectional)

        pitch_clipped = np.clip(pitch_values, 10, 170)
        interp = RegularGridInterpolator(
            (np.arange(y.shape[0]), np.arange(10, 180, 20)),
            y,
            bounds_error=False,
            fill_value=None,
        )
        y_interpolated = interp((np.arange(y.shape[0]), pitch_clipped))
        return self._to_real_scale(y_interpolated)
