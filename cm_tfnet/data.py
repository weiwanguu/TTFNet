"""数据集：滑窗预测下一帧双指 Fz（仅抬升起点→完全出水）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as cfg
from .medium_state import (
    build_medium_state_for_trial,
    finger_force_max,
    load_object_geometry,
)


FEATURE_DIM = 15  # 7L + 7R + s
FZ_L_IDX = 2
FZ_R_IDX = 9


def apply_fz_scale(X: np.ndarray, Y: np.ndarray, fz_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """仅缩放左右指 |Fz| 通道与标签。"""
    if abs(float(fz_scale) - 1.0) < 1e-12:
        return X, Y
    X = X.copy()
    Y = Y.copy()
    X[:, FZ_L_IDX] *= fz_scale
    X[:, FZ_R_IDX] *= fz_scale
    Y *= fz_scale
    return X, Y


TRIAL_COLUMNS = (
    "time_s",
    "L_fx1",
    "L_fy1",
    "L_fz1",
    "L_fx2",
    "L_fy2",
    "L_fz2",
    "L_proximity",
    "L_roll",
    "L_pitch",
    "L_yaw",
    "R_fx1",
    "R_fy1",
    "R_fz1",
    "R_fx2",
    "R_fy2",
    "R_fz2",
    "R_proximity",
    "R_roll",
    "R_pitch",
    "R_yaw",
)


def load_trial_frame(path: Path) -> "pd.DataFrame":
    """读取一次试验。npz 的键是各传感器通道。"""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"试验文件应为 .npz，收到 {path.name}")
    with np.load(path) as z:
        missing = [k for k in TRIAL_COLUMNS if k not in z.files]
        if missing:
            raise KeyError(f"{path} 缺少字段 {missing}")
        data = {k: np.asarray(z[k], dtype=np.float64) for k in TRIAL_COLUMNS}
    return pd.DataFrame(data)


def _norm_proximity(p: np.ndarray) -> np.ndarray:
    """接近信号已按 /65535 归一化到 [0,1]，此处直接读取。"""
    return np.asarray(p, dtype=np.float64)


@dataclass
class TrialMeta:
    folder: str
    file: str
    object_name: str
    t_peak: float
    t0: float
    H_m: float
    d_m: float
    n: int
    i_lift: int
    i_exit: int


def lift_exit_indices(
    time_s: np.ndarray,
    s: np.ndarray,
    t0: float,
    exit_extra_s: float | None = None,
) -> tuple[int, int]:
    """
    抬升起点帧 → 预测终点帧。
    默认终点为 s 首次达到 1（刚好完全出水）；
    exit_extra_s>0 时再向后延长若干秒。
    """
    t = np.asarray(time_s, dtype=np.float64)
    i_lift = int(np.searchsorted(t, t0, side="left"))
    i_lift = min(max(i_lift, 0), len(t) - 1)
    air = np.where(np.asarray(s) >= 1.0 - 1e-6)[0]
    if len(air):
        i_air = int(air[0])
    else:
        i_air = len(t) - 1
    extra = float(cfg.EXIT_EXTRA_S if exit_extra_s is None else exit_extra_s)
    if extra > 0:
        t_end = float(t[i_air]) + extra
        i_exit = int(np.searchsorted(t, t_end, side="left"))
        i_exit = min(i_exit, len(t) - 1)
    else:
        i_exit = i_air
    if i_exit < i_lift:
        i_exit = i_lift
    return i_lift, i_exit


def load_trial_arrays(
    trial_path: Path,
    H_m: float,
    d_m: float,
    exit_extra_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrialMeta]:
    """
    返回:
      X: (T, 15) 原始特征（未标准化）
      Y: (T, 2)  当前帧 Fz_L, Fz_R
      s: (T,)
      time_s: (T,)
      meta
    """
    df = load_trial_frame(trial_path)
    t = df["time_s"].to_numpy(dtype=np.float64)
    # 每指两组力：xyz 各自取绝对值更大者
    Lf = finger_force_max(df, "L")
    Rf = finger_force_max(df, "R")
    Lp = _norm_proximity(df["L_proximity"].to_numpy(dtype=np.float64))
    Rp = _norm_proximity(df["R_proximity"].to_numpy(dtype=np.float64))
    La = df[["L_roll", "L_pitch", "L_yaw"]].to_numpy(dtype=np.float64)
    Ra = df[["R_roll", "R_pitch", "R_yaw"]].to_numpy(dtype=np.float64)

    s, t_peak, t0 = build_medium_state_for_trial(df, H_m, d_m)
    i_lift, i_exit = lift_exit_indices(t, s, t0, exit_extra_s=exit_extra_s)

    left = np.concatenate([Lf, Lp[:, None], La], axis=1)  # (T,7)
    right = np.concatenate([Rf, Rp[:, None], Ra], axis=1)
    X = np.concatenate([left, right, s[:, None]], axis=1).astype(np.float32)
    Y = np.stack([np.abs(Lf[:, 2]), np.abs(Rf[:, 2])], axis=1).astype(np.float32)

    folder = trial_path.parent.name
    meta = TrialMeta(
        folder=folder,
        file=trial_path.name,
        object_name=folder,
        t_peak=float(t_peak),
        t0=float(t0),
        H_m=float(H_m),
        d_m=float(d_m),
        n=len(df),
        i_lift=i_lift,
        i_exit=i_exit,
    )
    return X, Y, s.astype(np.float32), t.astype(np.float64), meta


def discover_trials(data_root: Path | None = None) -> list[tuple[Path, dict]]:
    root = Path(data_root or cfg.DATA_ROOT)
    geom = load_object_geometry()
    trials = []
    for folder, g in sorted(geom.items()):
        d = root / folder
        if not d.is_dir():
            continue
        if "backup" in str(d).lower():
            continue
        for trial in sorted(d.glob("*.npz"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            trials.append((trial, g))
    return trials


def split_train_val_test(
    trials: list[tuple[Path, dict]],
    val_objects=cfg.VAL_OBJECTS,
    test_objects=cfg.TEST_OBJECTS,
) -> tuple[list, list, list]:
    """按物体划分 6:2:2 —— 指定验证/测试物体，其余训练。"""
    val_set = set(val_objects)
    test_set = set(test_objects)
    train, val, test = [], [], []
    for item in trials:
        name = item[0].parent.name
        if name in test_set:
            test.append(item)
        elif name in val_set:
            val.append(item)
        else:
            train.append(item)
    return train, val, test


def split_train_val_test_by_trial(
    trials: list[tuple[Path, dict]],
    seed: int = cfg.SEED,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[list, list, list, dict]:
    """
    每个物体的试次按 6:2:2 随机划分；保证每个物体都出现在 train/val/test。
    对 15 条试次：9 / 3 / 3。
    返回 train, val, test, split_detail。
    """
    rng = np.random.default_rng(seed)
    by_obj: dict[str, list[tuple[Path, dict]]] = {}
    for item in trials:
        by_obj.setdefault(item[0].parent.name, []).append(item)

    train, val, test = [], [], []
    split_detail: dict[str, dict[str, list[str]]] = {}

    for obj, items in sorted(by_obj.items()):
        items = sorted(items, key=lambda it: int(it[0].stem) if it[0].stem.isdigit() else it[0].stem)
        n = len(items)
        # 15 -> 9/3/3；一般 n*ratios 再把余数放 train
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        n_test = n - n_train - n_val
        if n >= 3:
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            n_train = n - n_val - n_test
            n_train = max(n_train, 1)

        idx = np.arange(n)
        rng.shuffle(idx)
        i_train = idx[:n_train]
        i_val = idx[n_train : n_train + n_val]
        i_test = idx[n_train + n_val :]

        tr = [items[i] for i in sorted(i_train)]
        va = [items[i] for i in sorted(i_val)]
        te = [items[i] for i in sorted(i_test)]
        train.extend(tr)
        val.extend(va)
        test.extend(te)
        split_detail[obj] = {
            "train": [p.name for p, _ in tr],
            "val": [p.name for p, _ in va],
            "test": [p.name for p, _ in te],
        }

    return train, val, test, split_detail


def trials_from_split_detail(
    trials: list[tuple[Path, dict]],
    split_detail: dict,
    which: str,
) -> list[tuple[Path, dict]]:
    """根据 split_detail 重建某一集合的 trial 列表。"""
    want = {}
    for obj, parts in split_detail.items():
        want[obj] = set(parts[which])
    out = []
    for item in trials:
        obj = item[0].parent.name
        if obj in want and item[0].name in want[obj]:
            out.append(item)
    return out


def compute_norm_stats(
    trials: list[tuple[Path, dict]],
    fz_scale: float = 1.0,
) -> dict:
    """仅用抬升→出水段上的帧估计统计量。"""
    xs = []
    ys = []
    for csv, g in trials:
        X, Y, s, t, meta = load_trial_arrays(csv, g["H_m"], g["d_m"])
        X, Y = apply_fz_scale(X, Y, fz_scale)
        seg = slice(meta.i_lift, meta.i_exit + 1)
        xs.append(X[seg])
        ys.append(Y[seg])
    Xall = np.concatenate(xs, axis=0)
    Yall = np.concatenate(ys, axis=0)
    return {
        "x_mean": Xall.mean(axis=0).tolist(),
        "x_std": (Xall.std(axis=0) + 1e-6).tolist(),
        "y_mean": Yall.mean(axis=0).tolist(),
        "y_std": (Yall.std(axis=0) + 1e-6).tolist(),
        "fz_scale": float(fz_scale),
    }


class GraspWindowDataset(Dataset):
    """
    单步监督 [t0, 完全出水] 内下一帧 Fz。
    输入窗可含 t0 前真实历史；标签只用段内帧。
    """

    def __init__(
        self,
        trials: list[tuple[Path, dict]],
        stats: dict,
        window: int = cfg.WINDOW,
        stride: int = cfg.STRIDE,
        horizon: int = cfg.HORIZON,
        fz_scale: float = 1.0,
    ):
        self.window = window
        self.stride = stride
        self.horizon = horizon
        self.fz_scale = float(fz_scale)
        self.x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(stats["x_std"], dtype=np.float32)

        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.metas: list[TrialMeta] = []

        for csv, g in trials:
            X, Y, s, t, meta = load_trial_arrays(csv, g["H_m"], g["d_m"])
            X, Y = apply_fz_scale(X, Y, self.fz_scale)
            Xn = (X - self.x_mean) / self.x_std
            i0, i1 = meta.i_lift, meta.i_exit
            for tgt_idx in range(i0, i1 + 1, stride):
                start = tgt_idx - window
                if start < 0:
                    continue
                x_win = Xn[start : start + window]
                y = Y[tgt_idx]
                w = 1.0 + (cfg.CROSS_MEDIUM_LOSS_WEIGHT - 1.0) * float(
                    (s[tgt_idx] > 0.0) and (s[tgt_idx] < 1.0)
                )
                self.samples.append((x_win, y, w))
            self.metas.append(meta)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, w = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(w, dtype=torch.float32),
        )


class GraspUnrollDataset(Dataset):
    """
    多步展开训练样本：返回 [t-W, t+K) 的归一化上下文与 K 步 Fz 标签。
    训练时用 scheduled sampling 把预测 Fz 写回上下文。
    """

    def __init__(
        self,
        trials: list[tuple[Path, dict]],
        stats: dict,
        window: int = cfg.WINDOW,
        unroll_steps: int = cfg.UNROLL_STEPS,
        stride: int = 4,
        fz_scale: float = 1.0,
    ):
        self.window = window
        self.unroll_steps = int(unroll_steps)
        self.stride = int(stride)
        self.fz_scale = float(fz_scale)
        self.x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(stats["x_std"], dtype=np.float32)

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.metas: list[TrialMeta] = []

        K = self.unroll_steps
        for csv, g in trials:
            X, Y, s, t, meta = load_trial_arrays(csv, g["H_m"], g["d_m"])
            X, Y = apply_fz_scale(X, Y, self.fz_scale)
            Xn = (X - self.x_mean) / self.x_std
            i0, i1 = meta.i_lift, meta.i_exit
            # 需要完整 K 步均落在监督段内，且有 W 帧历史
            last_start = i1 - K + 1
            if last_start < i0:
                continue
            for tgt_start in range(i0, last_start + 1, self.stride):
                if tgt_start < window:
                    continue
                x_ctx = Xn[tgt_start - window : tgt_start + K].astype(np.float32)  # (W+K, 15)
                y_seq = Y[tgt_start : tgt_start + K].astype(np.float32)  # (K, 2)
                w_seq = np.array(
                    [
                        1.0
                        + (cfg.CROSS_MEDIUM_LOSS_WEIGHT - 1.0)
                        * float((s[tgt_start + k] > 0.0) and (s[tgt_start + k] < 1.0))
                        for k in range(K)
                    ],
                    dtype=np.float32,
                )
                self.samples.append((x_ctx, y_seq, w_seq))
            self.metas.append(meta)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, w = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)



def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
