"""测试集自回归滚动评估：预测 Fz 回写窗口，其余与 onestep 设定相同。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as cfg
from .data import (
    discover_trials,
    load_trial_arrays,
    split_train_val_test,
    trials_from_split_detail,
)
from .infer_rollout import apply_fz_scale, rollout_trial
from .baselines import build_model


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae_L": float(np.mean(np.abs(err[:, 0]))),
        "mae_R": float(np.mean(np.abs(err[:, 1]))),
        "mse_L": float(np.mean(err[:, 0] ** 2)),
        "mse_R": float(np.mean(err[:, 1] ** 2)),
    }


def plot_segment(time_s, gt, pred, s, i0, i1, title, out_path, m):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = time_s
    ax[0].plot(t, gt[:, 0], label="GT Fz_L", lw=1.5)
    ax[0].plot(t, pred[:, 0], label="Pred Fz_L", lw=1.2, alpha=0.85)
    ax[0].axvspan(t[i0], t[i1], color="orange", alpha=0.15, label="eval seg")
    ax[0].set_ylabel("Fz_L")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, gt[:, 1], label="GT Fz_R", lw=1.5)
    ax[1].plot(t, pred[:, 1], label="Pred Fz_R", lw=1.2, alpha=0.85)
    ax[1].axvspan(t[i0], t[i1], color="orange", alpha=0.15)
    ax[1].set_ylabel("Fz_R")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, s, color="green", label="s(t)")
    ax[2].axhline(0, color="gray", lw=0.6)
    ax[2].axhline(1, color="gray", lw=0.6)
    ax[2].axvspan(t[i0], t[i1], color="orange", alpha=0.15)
    ax[2].set_ylabel("s(t)")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"{title}  [autoregressive rollout]\n"
        f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  RMSE={m['rmse']:.4f}  "
        f"(L={m['mae_L']:.4f}, R={m['mae_R']:.4f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(cfg.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default=str(cfg.CHECKPOINT_DIR / "test_eval_rollout"))
    parser.add_argument("--max-plots-per-obj", type=int, default=5)
    parser.add_argument(
        "--exit-extra",
        type=float,
        default=None,
        help="出水后再延长秒数；默认读 checkpoint.config.exit_extra_s",
    )
    parser.add_argument(
        "--fz-scale",
        type=float,
        default=None,
        help="Fz 缩放；默认读 checkpoint.config.fz_scale，缺省 1.0",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    window = ckpt.get("config", {}).get("window", cfg.WINDOW)
    exit_extra = (
        float(args.exit_extra)
        if args.exit_extra is not None
        else float(ckpt.get("config", {}).get("exit_extra_s", cfg.EXIT_EXTRA_S))
    )
    cfg.EXIT_EXTRA_S = exit_extra
    fz_scale = (
        float(args.fz_scale)
        if args.fz_scale is not None
        else float(ckpt.get("config", {}).get("fz_scale", 1.0))
    )
    split_mode = ckpt.get("split_mode", ckpt.get("config", {}).get("split_mode", "object"))
    split_detail = ckpt.get("split_detail")
    model_name = ckpt.get("config", {}).get("model", "cm_tfnet")
    d_model = int(ckpt.get("config", {}).get("d_model", cfg.D_MODEL))
    gru_hidden = int(ckpt.get("config", {}).get("gru_hidden", cfg.GRU_HIDDEN))
    dropout = float(ckpt.get("config", {}).get("dropout", cfg.DROPOUT))
    print(
        f"eval rollout  model={model_name}  exit_extra_s={exit_extra}  "
        f"split_mode={split_mode}  fz_scale={fz_scale}"
    )

    match_capacity = bool(ckpt.get("config", {}).get("match_capacity", False))
    model = build_model(
        model_name,
        d_model,
        gru_hidden,
        dropout,
        match_capacity=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  rebuilt model={model_name} d_model={d_model} hidden={gru_hidden} match_capacity_flag={match_capacity}")

    all_trials = discover_trials()
    if split_mode == "trial" and split_detail:
        test_trials = trials_from_split_detail(all_trials, split_detail, "test")
    else:
        _, _, test_trials = split_train_val_test(all_trials)
    print(f"test trials: {len(test_trials)}")

    all_rows = []
    by_obj: dict[str, list] = {}
    plot_count: dict[str, int] = {}
    preds_all, gts_all = [], []

    for csv, g in test_trials:
        X, Y, s, time_s, meta = load_trial_arrays(
            csv, g["H_m"], g["d_m"], exit_extra_s=exit_extra
        )
        X, Y = apply_fz_scale(X, Y, fz_scale)
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
        i0, i1 = out["start_idx"], out["end_idx"]
        pred = out["pred_fz"]
        seg = slice(i0, i1 + 1)
        m = metrics(pred[seg], Y[seg])
        row = {"object": meta.folder, "file": meta.file, "i_lift": i0, "i_exit": i1, **m}
        all_rows.append(row)
        by_obj.setdefault(meta.folder, []).append(m)
        preds_all.append(pred[seg])
        gts_all.append(Y[seg])

        (out_dir / meta.folder).mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / meta.folder / f"{Path(meta.file).stem}.npz",
            time_s=time_s,
            gt_fz=Y,
            pred_fz=pred,
            s=s,
            i_lift=i0,
            i_exit=i1,
            **m,
        )
        pc = plot_count.get(meta.folder, 0)
        if pc < args.max_plots_per_obj:
            plot_segment(
                time_s,
                Y,
                pred,
                s,
                i0,
                i1,
                f"{meta.folder}/{meta.file}",
                out_dir / meta.folder / f"{Path(meta.file).stem}.png",
                m,
            )
            plot_count[meta.folder] = pc + 1

    overall = metrics(np.concatenate(preds_all), np.concatenate(gts_all))
    lines = [
        "=== Test autoregressive rollout, lift->exit ===",
        f"fz_scale={fz_scale}  exit_extra_s={exit_extra}  split_mode={split_mode}",
        f"OVERALL  MAE={overall['mae']:.4f}  MSE={overall['mse']:.4f}  RMSE={overall['rmse']:.4f}  "
        f"MAE_L={overall['mae_L']:.4f}  MAE_R={overall['mae_R']:.4f}",
    ]
    for obj, ms in sorted(by_obj.items()):
        lines.append(
            f"{obj:24s}  n={len(ms):2d}  "
            f"MAE={np.mean([x['mae'] for x in ms]):.4f}±{np.std([x['mae'] for x in ms]):.4f}  "
            f"MSE={np.mean([x['mse'] for x in ms]):.4f}±{np.std([x['mse'] for x in ms]):.4f}  "
            f"RMSE={np.mean([x['rmse'] for x in ms]):.4f}"
        )
    lines.append("\n--- per trial ---")
    for r in all_rows:
        lines.append(
            f"{r['object']}/{r['file']}: MAE={r['mae']:.4f} MSE={r['mse']:.4f} "
            f"RMSE={r['rmse']:.4f} L={r['mae_L']:.4f} R={r['mae_R']:.4f}"
        )
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)

    for obj in sorted(by_obj):
        rows = [r for r in all_rows if r["object"] == obj]
        rows.sort(
            key=lambda r: int(Path(r["file"]).stem) if Path(r["file"]).stem.isdigit() else r["file"]
        )
        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(12, 2.1 * n), constrained_layout=True)
        if n == 1:
            axes = np.array([axes])
        for i, r in enumerate(rows):
            z = np.load(out_dir / obj / f"{Path(r['file']).stem}.npz")
            t = z["time_s"]
            i0, i1 = int(z["i_lift"]), int(z["i_exit"])
            seg = slice(i0, i1 + 1)
            axes[i, 0].plot(t[seg], z["gt_fz"][seg, 0], label="GT", lw=1.0)
            axes[i, 0].plot(t[seg], z["pred_fz"][seg, 0], label="Pred", alpha=0.85, lw=1.0)
            axes[i, 0].set_title(f"{r['file']} Fz_L MAE={r['mae_L']:.3f}", fontsize=9)
            axes[i, 0].legend(fontsize=7, loc="upper right")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(t[seg], z["gt_fz"][seg, 1], label="GT", lw=1.0)
            axes[i, 1].plot(t[seg], z["pred_fz"][seg, 1], label="Pred", alpha=0.85, lw=1.0)
            axes[i, 1].set_title(f"{r['file']} Fz_R MAE={r['mae_R']:.3f}", fontsize=9)
            axes[i, 1].legend(fontsize=7, loc="upper right")
            axes[i, 1].grid(True, alpha=0.3)
        fig.suptitle(f"{obj} autoregressive rollout — all {n} test trials", fontsize=12)
        fig.savefig(out_dir / f"{obj}_segment_compare.png", dpi=140)
        plt.close(fig)

    print(f"\nfigures -> {out_dir}")


if __name__ == "__main__":
    main()
