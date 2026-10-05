"""公平对比基线：扁平 15 维序列 -> 骨干 -> Softplus 双指 Fz（与 CM-TFNet 同 I/O）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import CMTFNet, TemporalBackbone

# 与 CM-TFNet(d=64,h=64)≈377030 参数对齐的预设宽度（2 层 RNN / 标准 TCN）
CAPACITY_MATCH_WIDTH = {
    "lstm": 148,  # ~377550
    "gru": 169,  # ~376534
    "tcn": 153,  # ~378524
    "cnn_gru": 143,  # ~376950
    "cnn_gru_tcn": 111,  # ~377180
    "cnn_bigru": 97,  # ~374422
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _pred_head(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
        nn.Softplus(),
    )


class LSTMBaseline(nn.Module):
    """(B,T,15) -> (B,2)"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


class GRUBaseline(nn.Module):
    """(B,T,15) -> (B,2)"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBaseline(nn.Module):
    """因果膨胀卷积 TCN，感受野覆盖窗长；取最后时刻预测。"""

    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(15, d_model)
        layers = []
        for dil in (1, 2, 4, 8, 16):
            layers.append(
                nn.Sequential(
                    CausalConv1d(d_model, d_model, kernel_size=3, dilation=dil),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = _pred_head(d_model, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.input_proj(x)
        y = h.transpose(1, 2)
        for blk in self.blocks:
            y = y + blk(y)
        h = self.norm(y.transpose(1, 2))
        out = self.head(h[:, -1, :])
        if return_gate:
            return out, None
        return out


class CNNGRUBaseline(nn.Module):
    """扁平 15 维：Conv1d（同 ModalityEncoder）+ 2 层 GRU。"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(15, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


class CNNGRUTCNBaseline(nn.Module):
    """扁平 15 维：Conv1d + 2 层 GRU + TemporalBackbone（同 CM-TFNet 积木，无分模态/门控）。"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(15, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.backbone = TemporalBackbone(hidden, dropout)
        self.head = _pred_head(hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        h = self.backbone(h)
        y = self.head(h[:, -1, :])
        if return_gate:
            return y, None
        return y


class CNNBiGRUBaseline(nn.Module):
    """扁平 15 维：Conv1d + 2 层双向 GRU（对齐 ModalityEncoder 的 BiGRU，但不分模态）。"""

    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(15, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _pred_head(2 * hidden, dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        y = self.head(out[:, -1, :])
        if return_gate:
            return y, None
        return y


def _actual_widths(model: nn.Module) -> tuple[int, int]:
    if hasattr(model, "gru") and hasattr(model, "conv"):
        return int(model.conv[0].out_channels), int(model.gru.hidden_size)
    if hasattr(model, "rnn") and hasattr(model, "input_proj"):
        return int(model.input_proj.out_features), int(model.rnn.hidden_size)
    if hasattr(model, "input_proj"):
        w = int(model.input_proj.out_features)
        return w, w
    return 64, 64


def build_model(
    name: str,
    d_model: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.1,
    num_layers: int = 2,
    match_capacity: bool = False,
) -> nn.Module:
    from .ablations import ABLATION_ALIASES, build_ablation

    key = name.strip().lower().replace("-", "_")
    if key in ("cnngru",):
        key = "cnn_gru"
    if key in ("cnngrutcn",):
        key = "cnn_gru_tcn"
    if key in ("cnnbigru", "cnn_bi_gru"):
        key = "cnn_bigru"
    if key in ("cm_tfnet", "cmtfnet", "ours"):
        return CMTFNet(d_model, gru_hidden, dropout)

    if key in ABLATION_ALIASES:
        return build_ablation(key, d_model, gru_hidden, dropout)

    if match_capacity and key in CAPACITY_MATCH_WIDTH:
        w = CAPACITY_MATCH_WIDTH[key]
        d_model = w
        gru_hidden = w

    if key == "lstm":
        return LSTMBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "gru":
        return GRUBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "tcn":
        return TCNBaseline(d_model, dropout)
    if key == "cnn_gru":
        return CNNGRUBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "cnn_gru_tcn":
        return CNNGRUTCNBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    if key == "cnn_bigru":
        return CNNBiGRUBaseline(d_model, gru_hidden, dropout, num_layers=num_layers)
    raise ValueError(f"unknown model: {name}")
