from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT

# 物理参数
WATER_DEPTH_M = 0.085  # 8.5 cm
SOFT_VEL = 0.03  # m/s
SOFT_DIST = 0.03  # m
SOFT_DURATION = SOFT_DIST / SOFT_VEL  # 1.0 s
CRUISE_VEL = 0.1  # m/s
PEAK_TO_LIFT_DELAY = 1.0  # s
# 预测终点相对“刚好完全出水”再延长的秒数；0=刚好出水，1=出水后再预测 1s
EXIT_EXTRA_S = 0.0

# 序列
# 窗长：预测 t0 附近 Fz 时，窗内可含 t0 前真实观测作上下文（与在线滚动一致）
WINDOW = 64
# 抬升→出水段较短，stride=1 加密监督
STRIDE = 1
HORIZON = 1  # 预测下一帧

# 模型
D_MODEL = 64
GRU_HIDDEN = 64
DROPOUT = 0.1

# 训练
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
SEED = 42
CROSS_MEDIUM_LOSS_WEIGHT = 3.0
SS_START = 0.2
SS_END = 0.85
LAMBDA_SMOOTH = 0.05
UNROLL_STEPS = 12
LAMBDA_DELTA = 0.5
CHECKPOINT_NAME = "best_v2.pt"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 物体划分 6:2:2（其余为训练集）
VAL_OBJECTS = ("grasp_logs_mosha", "grasp_logs_mosilian")
TEST_OBJECTS = ("grasp_logs_qiu", "grasp_logs_taocibei")
