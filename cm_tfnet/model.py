"""CM-TFNet: 单步预测下一帧绝对 Fz（Softplus）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * gru_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.drop(self.proj(h))


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, n_mod: int = 4):
        super().__init__()
        self.gate = nn.Linear(d_model * n_mod, n_mod)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(feats, dim=-1)
        g = torch.softmax(self.gate(cat), dim=-1)
        stacked = torch.stack(feats, dim=-1)
        fused = (stacked * g.unsqueeze(2)).sum(dim=-1)
        return self.out(fused), g


class TemporalBackbone(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        layers = []
        for dil in (1, 2, 4, 8):
            layers += [
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dil, dilation=dil),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class CMTFNet(nn.Module):
    """输入窗 (B,T,15) -> 下一帧 [Fz_L, Fz_R]。"""

    def __init__(self, d_model: int = 64, gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.force_enc = ModalityEncoder(6, d_model, gru_hidden, dropout)
        self.prox_enc = ModalityEncoder(2, d_model, gru_hidden, dropout)
        self.imu_enc = ModalityEncoder(6, d_model, gru_hidden, dropout)
        self.med_enc = ModalityEncoder(1, d_model, gru_hidden, dropout)
        self.fusion = GatedFusion(d_model, n_mod=4)
        self.backbone = TemporalBackbone(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Softplus(),
        )

    def split_modalities(self, x: torch.Tensor):
        force = torch.cat([x[..., 0:3], x[..., 7:10]], dim=-1)
        prox = torch.cat([x[..., 3:4], x[..., 10:11]], dim=-1)
        imu = torch.cat([x[..., 4:7], x[..., 11:14]], dim=-1)
        med = x[..., 14:15]
        return force, prox, imu, med

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        force, prox, imu, med = self.split_modalities(x)
        fused, gate = self.fusion(
            [self.force_enc(force), self.prox_enc(prox), self.imu_enc(imu), self.med_enc(med)]
        )
        h = self.backbone(fused)
        y = self.head(h[:, -1, :])
        if return_gate:
            return y, gate
        return y
