"""CM-TFNet 滚动训练：scheduled sampling + multi-step unroll。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .data import (
    FZ_L_IDX,
    FZ_R_IDX,
    GraspUnrollDataset,
    GraspWindowDataset,
    compute_norm_stats,
    discover_trials,
    save_json,
    split_train_val_test,
    split_train_val_test_by_trial,
)
from .baselines import build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ss_use_pred_prob(epoch: int, epochs: int, start: float, end: float) -> float:
    """线性增大「使用模型预测写回」的概率。"""
    if epochs <= 1:
        return end
    t = epoch / (epochs - 1)
    return float(start + (end - start) * t)


def _write_fz_norm(buf: torch.Tensor, idx: int, fz: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """把绝对 Fz 写入归一化上下文 buf[:, idx, Fz]。"""
    buf[:, idx, FZ_L_IDX] = (fz[:, 0] - mean[FZ_L_IDX]) / std[FZ_L_IDX]
    buf[:, idx, FZ_R_IDX] = (fz[:, 1] - mean[FZ_R_IDX]) / std[FZ_R_IDX]


def unroll_forward(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    y_seq: torch.Tensor,
    w_seq: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    window: int,
    p_use_pred: float,
    train_mode: bool,
    lambda_smooth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x_ctx: (B, W+K, 15) 归一化；y_seq: (B, K, 2)；w_seq: (B, K)
    返回 loss, mae（按步平均）。
    """
    B, WK, _ = x_ctx.shape
    K = y_seq.size(1)
    assert WK == window + K
    buf = x_ctx.clone()
    step_losses = []
    step_maes = []
    prev_pred = None

    for k in range(K):
        win = buf[:, k : k + window, :].clone()
        pred = model(win)
        mae_k = (pred - y_seq[:, k]).abs().mean(dim=1)
        mse_k = ((pred - y_seq[:, k]) ** 2).mean(dim=1)
        loss_k = w_seq[:, k] * (mae_k + mse_k)
        if prev_pred is not None and lambda_smooth > 0:
            loss_k = loss_k + lambda_smooth * (pred - prev_pred.detach()).abs().mean(dim=1)
        step_losses.append(loss_k)
        step_maes.append(mae_k)
        prev_pred = pred

        # 本步 Fz 写回窗口，下一步只看到写回后的力
        write_idx = window + k
        if write_idx >= WK:
            continue
        if train_mode and p_use_pred > 0:
            use_pred = torch.rand(B, device=x_ctx.device) < p_use_pred
            fz_write = torch.where(use_pred.unsqueeze(1), pred.detach(), y_seq[:, k])
        elif (not train_mode) or p_use_pred >= 1.0:
            # 验证/纯滚动：一律用预测
            fz_write = pred.detach()
        else:
            fz_write = y_seq[:, k]
        _write_fz_norm(buf, write_idx, fz_write, mean, std)

    loss = torch.stack(step_losses, dim=1).mean()
    mae = torch.stack(step_maes, dim=1).mean()
    return loss, mae


def train_one_epoch(model, loader, opt, device, mean, std, window, p_use_pred, lambda_smooth) -> dict:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for x_ctx, y_seq, w_seq in loader:
        x_ctx = x_ctx.to(device)
        y_seq = y_seq.to(device)
        w_seq = w_seq.to(device)
        opt.zero_grad(set_to_none=True)
        loss, mae = unroll_forward(
            model,
            x_ctx,
            y_seq,
            w_seq,
            mean,
            std,
            window,
            p_use_pred=p_use_pred,
            train_mode=True,
            lambda_smooth=lambda_smooth,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = x_ctx.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(mae.item()) * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "mae": total_mae / max(n, 1)}


@torch.no_grad()
def eval_unroll(model, loader, device, mean, std, window, lambda_smooth) -> dict:
    """验证：强制自回归写回（p_use_pred=1）。"""
    model.eval()
    total_mae = 0.0
    total_loss = 0.0
    n = 0
    for x_ctx, y_seq, w_seq in loader:
        x_ctx = x_ctx.to(device)
        y_seq = y_seq.to(device)
        w_seq = w_seq.to(device)
        loss, mae = unroll_forward(
            model,
            x_ctx,
            y_seq,
            w_seq,
            mean,
            std,
            window,
            p_use_pred=1.0,
            train_mode=False,
            lambda_smooth=lambda_smooth,
        )
        bs = x_ctx.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(mae.item()) * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "mae": total_mae / max(n, 1)}


@torch.no_grad()
def eval_onestep(model, loader, device) -> dict:
    model.eval()
    total_mae = 0.0
    n = 0
    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        mae = (pred - y).abs().mean(dim=1)
        bs = x.size(0)
        total_mae += float(mae.mean().item()) * bs
        n += bs
    return {"mae": total_mae / max(n, 1)}


def _count_objects(trials) -> list[str]:
    return sorted({p.parent.name for p, _ in trials})


def _count_files(trials) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p, _ in trials:
        out.setdefault(p.parent.name, []).append(p.name)
    for k in out:
        out[k] = sorted(out[k], key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train CM-TFNet with SS + unroll")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exit-extra", type=float, default=1.0)
    parser.add_argument("--fz-scale", type=float, default=1.0)
    parser.add_argument("--split-mode", type=str, default="trial", choices=("object", "trial"))
    parser.add_argument("--split-seed", type=int, default=cfg.SEED)
    parser.add_argument("--unroll-steps", type=int, default=cfg.UNROLL_STEPS)
    parser.add_argument("--unroll-stride", type=int, default=4)
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="每隔多少个 epoch 做一次验证/存盘（最后一轮必验证）",
    )
    parser.add_argument(
        "--val-stride",
        type=int,
        default=0,
        help="验证集 unroll 采样步长；0 表示与 --unroll-stride 相同，可设更大以加速验证",
    )
    parser.add_argument("--ss-start", type=float, default=cfg.SS_START)
    parser.add_argument("--ss-end", type=float, default=cfg.SS_END)
    parser.add_argument("--lambda-smooth", type=float, default=cfg.LAMBDA_SMOOTH)
    parser.add_argument("--ckpt-name", type=str, default="best_plus1s_trialsplit_fz2_ss.pt")
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="可选：从已有单步权重热启动，如 best_plus1s_trialsplit_fz2.pt",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="从中断的 SS 训练 ckpt 续训（恢复 epoch+1 并沿用 SS 课表）",
    )
    parser.add_argument(
        "--exclude-objects",
        type=str,
        default="",
        help="排除的物体文件夹，逗号分隔，如 grasp_logs_qiu,grasp_logs_suliaoping",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cm_tfnet",
        choices=(
            "cm_tfnet",
            "lstm",
            "gru",
            "tcn",
            "cnn_gru",
            "cnn_gru_tcn",
            "cnn_bigru",
            "no_gate",
            "no_multimodal",
            "no_backbone",
            "no_med",
            "no_prox",
            "no_imu",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5a",
            "a5b",
        ),
        help="cm_tfnet / 基线 lstm|gru|tcn / 消融 no_gate|no_multimodal|no_backbone|no_med|no_prox|no_imu",
    )
    parser.add_argument(
        "--match-capacity",
        action="store_true",
        help="基线加宽到与 CM-TFNet(~377k) 参数量接近",
    )
    parser.add_argument("--d-model", type=int, default=0, help="覆盖 d_model；0=用 config 或 match-capacity")
    parser.add_argument("--hidden", type=int, default=0, help="覆盖 RNN hidden；0=同 d_model/config")
    args = parser.parse_args()
    if args.val_every < 1:
        args.val_every = 1
    if args.val_stride <= 0:
        args.val_stride = args.unroll_stride

    cfg.EXIT_EXTRA_S = float(args.exit_extra)
    set_seed(cfg.SEED)
    device = torch.device(args.device)
    fz_scale = float(args.fz_scale)
    window = cfg.WINDOW
    exclude: set[str] = set()
    for x in args.exclude_objects.split(","):
        x = x.strip()
        if not x:
            continue
        exclude.add(x if x.startswith("grasp_logs_") else f"grasp_logs_{x}")

    trials = discover_trials()
    if exclude:
        before = len(trials)
        trials = [it for it in trials if it[0].parent.name not in exclude]
        print(f"  exclude {sorted(exclude)}: trials {before} -> {len(trials)}")
    split_detail = None
    if args.split_mode == "trial":
        train_trials, val_trials, test_trials, split_detail = split_train_val_test_by_trial(
            trials, seed=args.split_seed
        )
    else:
        train_trials, val_trials, test_trials = split_train_val_test(trials)

    print(f"{args.model} rollout train (scheduled sampling + multi-step unroll)")
    print(
        f"  model={args.model}  exit_extra_s={cfg.EXIT_EXTRA_S}  split_mode={args.split_mode}  "
        f"split_seed={args.split_seed}  fz_scale={fz_scale}  K={args.unroll_steps}  "
        f"SS=[{args.ss_start},{args.ss_end}]  val_every={args.val_every}  val_stride={args.val_stride}"
    )
    print(f"  train ({len(train_trials)}): {_count_objects(train_trials)}")
    print(f"  val   ({len(val_trials)}): {_count_objects(val_trials)}")
    print(f"  test  ({len(test_trials)}): {_count_objects(test_trials)}")

    stats = compute_norm_stats(train_trials, fz_scale=fz_scale)
    tag = f"_plus{cfg.EXIT_EXTRA_S:g}s"
    if args.split_mode == "trial":
        tag += "_trialsplit"
    if abs(fz_scale - 1.0) > 1e-12:
        tag += f"_fz{fz_scale:g}"
    tag += "_ss"
    save_json(stats, cfg.CHECKPOINT_DIR / f"norm_stats{tag}.json")
    if split_detail is not None:
        save_json(split_detail, cfg.CHECKPOINT_DIR / f"split_detail{tag}.json")

    train_ds = GraspUnrollDataset(
        train_trials,
        stats,
        window=window,
        unroll_steps=args.unroll_steps,
        stride=args.unroll_stride,
        fz_scale=fz_scale,
    )
    val_ds = GraspUnrollDataset(
        val_trials,
        stats,
        window=window,
        unroll_steps=args.unroll_steps,
        stride=args.val_stride,
        fz_scale=fz_scale,
    )
    # 单步窗口只作对照，正式指标用滚动预测
    val_one_ds = GraspWindowDataset(val_trials, stats, fz_scale=fz_scale)
    test_one_ds = GraspWindowDataset(test_trials, stats, fz_scale=fz_scale)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_one_loader = DataLoader(val_one_ds, batch_size=64, shuffle=False, num_workers=0)

    d_model = int(args.d_model) if args.d_model > 0 else cfg.D_MODEL
    hidden = int(args.hidden) if args.hidden > 0 else (d_model if args.d_model > 0 else cfg.GRU_HIDDEN)
    model = build_model(
        args.model,
        d_model=d_model,
        gru_hidden=hidden,
        dropout=cfg.DROPOUT,
        match_capacity=bool(args.match_capacity),
    ).to(device)
    if args.match_capacity and args.model in (
        "lstm",
        "gru",
        "tcn",
        "cnn_gru",
        "cnn_gru_tcn",
        "cnn_bigru",
    ):
        from .baselines import _actual_widths

        d_model, hidden = _actual_widths(model)
    start_epoch = 0
    best = float("inf")
    history = []
    ckpt_path = cfg.CHECKPOINT_DIR / args.ckpt_name
    n_params = count_params(model)
    print(f"  params={n_params:,}  d_model={d_model}  hidden={hidden}  match_capacity={args.match_capacity}")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = cfg.CHECKPOINT_DIR / resume_path
        ck_r = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck_r["model"], strict=True)
        start_epoch = int(ck_r.get("epoch", -1)) + 1
        best = float(ck_r.get("val_unroll_mae", ck_r.get("val_mae", float("inf"))))
        if ck_r.get("stats") is not None:
            stats = ck_r["stats"]
        if ck_r.get("split_detail") is not None:
            split_detail = ck_r["split_detail"]
        print(f"  resume from {resume_path} @ epoch {start_epoch}  best_val={best:.4f}")
    elif args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.is_absolute():
            init_path = cfg.CHECKPOINT_DIR / init_path
        ck0 = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(ck0["model"], strict=True)
        print(f"  warm-start from {init_path}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for _ in range(start_epoch):
        sched.step()

    mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)

    print(
        f"unroll samples train={len(train_ds)} val={len(val_ds)}  "
        f"onestep val={len(val_one_ds)} device={device}"
    )

    for epoch in range(start_epoch, args.epochs):
        p_pred = ss_use_pred_prob(epoch, args.epochs, args.ss_start, args.ss_end)
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            mean,
            std,
            window,
            p_use_pred=p_pred,
            lambda_smooth=args.lambda_smooth,
        )
        sched.step()
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch + 1 == args.epochs)
        row = {
            "epoch": epoch,
            "p_use_pred": p_pred,
            **{f"train_{k}": v for k, v in tr.items()},
        }
        if do_val:
            va = eval_unroll(
                model, val_loader, device, mean, std, window, lambda_smooth=args.lambda_smooth
            )
            va1 = eval_onestep(model, val_one_loader, device)
            row.update({f"val_unroll_{k}": v for k, v in va.items()})
            row["val_onestep_mae"] = va1["mae"]
            print(
                f"[{epoch:03d}] p_pred={p_pred:.2f} train_loss={tr['loss']:.4f} train_mae={tr['mae']:.4f} "
                f"val_unroll_mae={va['mae']:.4f} val_onestep_mae={va1['mae']:.4f}"
            )
            if va["mae"] < best:
                best = va["mae"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stats": stats,
                        "epoch": epoch,
                        "val_mae": best,
                        "val_unroll_mae": best,
                        "split_mode": args.split_mode,
                        "split_seed": args.split_seed,
                        "split_detail": split_detail,
                        "split": {
                            "train": _count_files(train_trials),
                            "val": _count_files(val_trials),
                            "test": _count_files(test_trials),
                        },
                        "config": {
                            "model": args.model,
                            "window": window,
                            "stride": cfg.STRIDE,
                            "d_model": d_model,
                            "gru_hidden": hidden,
                            "dropout": cfg.DROPOUT,
                            "match_capacity": bool(args.match_capacity),
                            "force": "per_axis_max_abs",
                            "prox": "csv_/65535",
                            "fz_scale": fz_scale,
                            "exit_extra_s": cfg.EXIT_EXTRA_S,
                            "split_mode": args.split_mode,
                            "train_mode": "scheduled_sampling_unroll",
                            "unroll_steps": args.unroll_steps,
                            "ss_start": args.ss_start,
                            "ss_end": args.ss_end,
                            "lambda_smooth": args.lambda_smooth,
                            "val_every": args.val_every,
                            "exclude_objects": sorted(exclude),
                            "n_params": n_params,
                        },
                    },
                    ckpt_path,
                )
        else:
            print(
                f"[{epoch:03d}] p_pred={p_pred:.2f} train_loss={tr['loss']:.4f} "
                f"train_mae={tr['mae']:.4f}  (skip val)"
            )
        history.append(row)

    if not ckpt_path.exists() and args.resume:
        pass
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te = eval_unroll(
        model,
        DataLoader(
            GraspUnrollDataset(
                test_trials,
                stats,
                window=window,
                unroll_steps=args.unroll_steps,
                stride=args.unroll_stride,
                fz_scale=fz_scale,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        device,
        mean,
        std,
        window,
        lambda_smooth=args.lambda_smooth,
    )
    te1 = eval_onestep(
        model,
        DataLoader(test_one_ds, batch_size=64, shuffle=False, num_workers=0),
        device,
    )
    print(f"best val_unroll MAE={best:.4f} @ epoch {ckpt['epoch']}")
    print(f"test unroll MAE={te['mae']:.4f}  test onestep MAE={te1['mae']:.4f}")
    history.append({"final_test_unroll": te, "final_test_onestep": te1})
    hist_name = f"history{tag}"
    if args.model not in ("cm_tfnet", "cmtfnet"):
        hist_name += f"_{args.model}"
        if args.match_capacity:
            hist_name += "_capmatch"
    save_json(history, cfg.CHECKPOINT_DIR / f"{hist_name}.json")
    print(f"done -> {ckpt_path}")


if __name__ == "__main__":
    main()
