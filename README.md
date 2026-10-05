# X-MulTac-hub (Based on Self-Developed Cross-Medium Multimodal Tactile Sensor)

Cross-medium grasp force forecasting with **TTFNet** (Cross-Medium Temporal Fusion Network).

The network is a next-frame predictor: given a sliding multimodal window `(B, T, 15)`, it outputs `[Fz_L, Fz_R]`. The reported protocol is **autoregressive rollout**. During lift-to-exit, predicted Fz is written back into the window; proximity, IMU, and medium state `s(t)` stay observed. Training matches this with scheduled sampling and multi-step unroll.

This repository contains the **dataset** and **model code** (including ablations and baselines). 

## Layout

```
CMMTS-hub/
├── cm_tfnet/                 # model, training, evaluation, ablations, baselines
│   ├── model.py              # TTFNet
│   ├── ablations.py          # architectural ablations
│   ├── baselines.py          # LSTM / GRU / TCN baselines
│   ├── train_rollout.py      # rollout training (scheduled sampling + unroll)
│   ├── eval_rollout.py       # autoregressive rollout evaluation
│   └── ...
├── grasp_logs_*/             # grasp logs for 8 objects (15 trial NPZ files each)
```

Each trial is a 15-D sequence: left/right finger force, proximity, IMU, and the medium state `s(t)`.

## Setup

```bash
pip install -r cm_tfnet/requirements.txt
```

Dependencies: `torch>=2.0`, `numpy`, `pandas`, `openpyxl`, `matplotlib`.

Run the commands below from the repository root.

## Training

Full CM-TFNet (trial split; predict 1 s after water exit):

```bash
python -m cm_tfnet.train_rollout --model cm_tfnet --split-mode trial --split-seed 456 --exit-extra 1 --fz-scale 1 --ss-start 0.7 --ss-end 0.95
```

### Ablations

| Flag | Description |
|------|-------------|
| `no_gate` / `a1` | Mean fusion, no gating |
| `no_multimodal` / `a2` | Single 15-D encoder |
| `no_backbone` / `a3` | Remove TemporalBackbone |
| `no_med` / `a4` | Remove `s(t)` |
| `no_prox` / `a5a` | Remove proximity |
| `no_imu` / `a5b` | Remove IMU |

```bash
python -m cm_tfnet.train_rollout --model no_gate --split-mode trial --split-seed 456 --exit-extra 1 --fz-scale 1 --ss-start 0.7 --ss-end 0.95
```

### Baselines

```bash
python -m cm_tfnet.train_rollout --model lstm --match-capacity --split-mode trial --split-seed 456 --exit-extra 1 --fz-scale 1 --ss-start 0.7 --ss-end 0.95
python -m cm_tfnet.train_rollout --model gru --match-capacity --split-mode trial --split-seed 456 --exit-extra 1 --fz-scale 1 --ss-start 0.7 --ss-end 0.95
python -m cm_tfnet.train_rollout --model tcn --match-capacity --split-mode trial --split-seed 456 --exit-extra 1 --fz-scale 1 --ss-start 0.7 --ss-end 0.95
```

`--match-capacity` scales each baseline to roughly the same parameter count as TTFNet (~377k). The same flags apply to `cnn_gru`, `cnn_gru_tcn`, and `cnn_bigru`.

Released weights are in `checkpoints/`: the main model, six baselines, and six ablations. New training runs are also written there. Use `--fz-scale 1`; the force columns in the NPZ files are already in newtons.

## Evaluation

```bash
python -m cm_tfnet.eval_rollout --ckpt checkpoints/FINAL_best_plus1s_trialsplit_ss_seed456_noqiu_nosuliaoping_k12.pt
```

## Dataset

| Directory | Object |
|-----------|--------|
| `grasp_logs_beike` | Shell |
| `grasp_logs_mosha` | Frosted Glass Model |
| `grasp_logs_mosilian` | Yogurt Carton |
| `grasp_logs_paomo` | Sponge |
| `grasp_logs_ruanzhu` | Soft Silicone Model  |
| `grasp_logs_taocibei` | Ceramic cup |
| `grasp_logs_touming` | Smooth Glass Model |
| `grasp_logs_yingzhu` | Hard Silicone Model |
