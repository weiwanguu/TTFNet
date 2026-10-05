"""介质浮力损失状态 s(t) 与过冲回落检测。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from . import config as cfg


def load_object_geometry() -> dict[str, dict[str, float]]:
    """读取物体几何: folder -> {H_m, d_m}。"""
    import json

    json_path = Path(cfg.ROOT) / "cm_tfnet" / "object_geometry.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


def finger_force_max(df: pd.DataFrame, side: str) -> np.ndarray:
    """两组传感按分量取绝对值更大者，返回 (T, 3)=fx,fy,fz。"""
    axes = []
    for ax in ("fx", "fy", "fz"):
        a = df[f"{side}_{ax}1"].to_numpy(dtype=np.float64)
        b = df[f"{side}_{ax}2"].to_numpy(dtype=np.float64)
        pick = np.where(np.abs(a) >= np.abs(b), a, b)
        axes.append(pick)
    return np.stack(axes, axis=1)


finger_force_mean = finger_force_max
finger_force_by_max_fz = finger_force_max


def combined_fz(df: pd.DataFrame) -> np.ndarray:
    lf = finger_force_max(df, "L")[:, 2]
    rf = finger_force_max(df, "R")[:, 2]
    return np.maximum(np.abs(lf), np.abs(rf))


def detect_overshoot_fallback_time(
    fz: np.ndarray,
    time_s: np.ndarray,
    min_force: float = 0.05,
    smooth_win: int = 5,
    max_search_s: float = 8.0,
    min_drop_ratio: float = 0.08,
    confirm: int = 3,
) -> float:
    """
    接触后第一次明显过冲局部峰之后，力开始回落的第一个时刻。

    返回 time_s 中的时刻（秒）。
    """
    y = fz.astype(np.float64).copy()
    if smooth_win > 1:
        k = np.ones(smooth_win) / smooth_win
        y = np.convolve(y, k, mode="same")

    t = time_s.astype(np.float64)
    mask = t <= (t[0] + max_search_s)
    y_s = y[mask]
    t_s = t[mask]
    n = len(y_s)
    if n < 10:
        return float(t[int(np.argmax(fz))])

    # 接触起点：力首次超过阈值
    thr = max(min_force, 0.15 * float(np.max(y_s)))
    above = np.where(y_s > thr)[0]
    if len(above) == 0:
        return float(t[int(np.argmax(fz))])
    i_contact = int(above[0])

    # 在接触后寻找第一个“过冲峰 + 明显回落”
    i_peak = None
    for i in range(i_contact + 2, n - confirm - 1):
        if not (y_s[i] >= y_s[i - 1] and y_s[i] >= y_s[i + 1]):
            continue
        if y_s[i] < thr:
            continue
        # 峰前上升幅度
        left_min = float(np.min(y_s[i_contact : i + 1]))
        prom = y_s[i] - left_min
        if prom < max(min_force, 0.2 * y_s[i]):
            continue
        # 峰后确认回落
        future = y_s[i + 1 : i + 1 + max(8, confirm * 4)]
        if len(future) < confirm:
            continue
        drop = y_s[i] - float(np.min(future))
        if drop < min_drop_ratio * y_s[i]:
            continue
        # 连续 confirm 步总体下行
        if future[confirm - 1] < y_s[i]:
            i_peak = i
            break

    if i_peak is None:
        # 回退：搜索窗内最大峰
        i_peak = int(np.argmax(y_s[i_contact:])) + i_contact

    # 回落第一个时刻：峰后首次 y[i] < y[i-1]
    for j in range(i_peak + 1, n):
        if y_s[j] < y_s[j - 1]:
            return float(t_s[j])
    return float(t_s[i_peak])


def lift_displacement(dt: np.ndarray) -> np.ndarray:
    """抬升开始后经过 dt 秒的位移（m），两段速度。"""
    dt = np.asarray(dt, dtype=np.float64)
    h = np.zeros_like(dt)
    soft = dt <= cfg.SOFT_DURATION
    h[soft] = cfg.SOFT_VEL * np.maximum(dt[soft], 0.0)
    cruise = ~soft
    h[cruise] = cfg.SOFT_DIST + cfg.CRUISE_VEL * (dt[cruise] - cfg.SOFT_DURATION)
    h[dt < 0] = 0.0
    return h


def buoyancy_loss_state(
    time_s: np.ndarray,
    t0: float,
    H_m: float,
    d_m: float,
) -> np.ndarray:
    """
    s(t) = 1 - λ(t), λ = V_sub / V ≈ h_sub / H。

    水面 z=0，向上为正；抬升前顶面在 z=-d。
    """
    t = np.asarray(time_s, dtype=np.float64)
    h = lift_displacement(t - t0)
    z_top = -d_m + h
    z_bottom = z_top - H_m

    lam = np.empty_like(t)
    fully_under = z_top <= 0.0
    fully_air = z_bottom >= 0.0
    interface = ~(fully_under | fully_air)

    lam[fully_under] = 1.0
    lam[fully_air] = 0.0
    lam[interface] = np.clip(-z_bottom[interface] / max(H_m, 1e-8), 0.0, 1.0)
    return 1.0 - lam


def build_medium_state_for_trial(
    df: pd.DataFrame,
    H_m: float,
    d_m: float,
) -> tuple[np.ndarray, float, float]:
    """返回 s(t), t_peak(回落起点), t0。"""
    fz = combined_fz(df)
    t = df["time_s"].to_numpy(dtype=np.float64)
    t_peak = detect_overshoot_fallback_time(fz, t)
    t0 = t_peak + cfg.PEAK_TO_LIFT_DELAY
    s = buoyancy_loss_state(t, t0, H_m, d_m)
    return s, t_peak, t0
