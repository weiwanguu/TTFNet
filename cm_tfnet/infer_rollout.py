"""跨介质阶段自回归滚动推理：历史 Fz 回写预测值，其余模态用真实观测。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as cfg
from .data import FZ_L_IDX, FZ_R_IDX, apply_fz_scale, load_trial_arrays
from .medium_state import load_object_geometry
from .model import CMTFNet


@torch.no_grad()
def rollout_trial(
    model: CMTFNet,
    X: np.ndarray,
    Y: np.ndarray,
    s: np.ndarray,
    stats: dict,
    window: int,
    device: torch.device,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> dict:
    """
    X: (T,15) 原始特征, Y: (T,2) GT Fz
    预测区间 [start_idx, end_idx]：窗内 Fz 用已预测值回写；其余模态始终用真实观测。
    start_idx 之前的 Fz 保持真值（暖启动）。
    """
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)

    T = len(X)
    if start_idx is None or end_idx is None:
        raise ValueError("必须提供 i_lift/i_exit")
    end_idx = min(end_idx, T - 1)
    start_idx = max(int(start_idx), 0)
    if start_idx < window:
        start_idx = window

    pred_fz = Y.copy()
    for t in range(start_idx, end_idx + 1):
        sl = slice(t - window, t)
        xin = X[sl].copy()
        # 预测段内已走过的时刻：用预测 Fz 覆盖
        for k in range(sl.start, sl.stop):
            if k >= start_idx:
                xin[k - sl.start, FZ_L_IDX] = pred_fz[k, 0]
                xin[k - sl.start, FZ_R_IDX] = pred_fz[k, 1]
        xn = (xin - x_mean) / x_std
        xt = torch.from_numpy(xn.astype(np.float32)).unsqueeze(0).to(device)
        yhat = model(xt).cpu().numpy()[0]
        pred_fz[t] = yhat

    return {
        "pred_fz": pred_fz,
        "gt_fz": Y,
        "s": s,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(cfg.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--csv", type=str, required=True, help="试验 npz 路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fz-scale", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    window = ckpt.get("config", {}).get("window", cfg.WINDOW)
    exit_extra = float(ckpt.get("config", {}).get("exit_extra_s", cfg.EXIT_EXTRA_S))
    cfg.EXIT_EXTRA_S = exit_extra

    model = CMTFNet(cfg.D_MODEL, cfg.GRU_HIDDEN, cfg.DROPOUT).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = cfg.DATA_ROOT / csv_path
    geom = load_object_geometry()
    g = geom[csv_path.parent.name]
    X, Y, s, time_s, meta = load_trial_arrays(
        csv_path, g["H_m"], g["d_m"], exit_extra_s=exit_extra
    )
    X, Y = apply_fz_scale(X, Y, args.fz_scale)

    out = rollout_trial(
        model,
        X,
        Y,
        s,
        stats,
        window,
        device,
        start_idx=meta.i_lift,
        end_idx=meta.i_exit,
    )
    pred = out["pred_fz"]
    gt = out["gt_fz"]
    i0, i1 = out["start_idx"], out["end_idx"]
    seg = slice(i0, i1 + 1)
    mae = np.mean(np.abs(pred[seg] - gt[seg]))
    print(f"{csv_path}  lift->exit [{i0},{i1}]  MAE={mae:.4f}  fz_scale={args.fz_scale}")

    t = time_s
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, gt[:, 0], label="GT Fz_L")
    ax[0].plot(t, pred[:, 0], label="Pred Fz_L", alpha=0.8)
    ax[0].axvspan(t[i0], t[i1], color="orange", alpha=0.15, label="rollout")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("Fz_L")
    ax[1].plot(t, gt[:, 1], label="GT Fz_R")
    ax[1].plot(t, pred[:, 1], label="Pred Fz_R", alpha=0.8)
    ax[1].axvspan(t[i0], t[i1], color="orange", alpha=0.15)
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("Fz_R")
    ax[2].plot(t, s, label="s(t)")
    ax[2].axhline(0, color="gray", lw=0.5)
    ax[2].axhline(1, color="gray", lw=0.5)
    ax[2].axvspan(t[i0], t[i1], color="orange", alpha=0.15)
    ax[2].set_ylabel("s(t)")
    ax[2].set_xlabel("time (s)")
    fig.suptitle(f"{csv_path.parent.name}/{csv_path.name}  rollout MAE={mae:.4f}")
    fig.tight_layout()

    out_path = (
        Path(args.out)
        if args.out
        else cfg.CHECKPOINT_DIR / f"rollout_{csv_path.parent.name}_{csv_path.stem}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
