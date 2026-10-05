"""CM-TFNet 结构消融变体（同 I/O：(B,T,15)->(B,2) Softplus）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import CMTFNet, GatedFusion, ModalityEncoder, TemporalBackbone


class MeanFusion(nn.Module):
    """无门控：各模态特征等权平均后再线性投影。"""

    def __init__(self, d_model: int, n_mod: int = 4):
        super().__init__()
        self.n_mod = n_mod
        self.out = nn.Linear(d_model, d_model)

    def forward(self, feats: list[torch.Tensor]):
        stacked = torch.stack(feats, dim=-1)
        fused = stacked.mean(dim=-1)
        return self.out(fused), None


class CMTFNetAblation(nn.Module):
    """
    ablation:
      full          - 完整 CM-TFNet
      no_gate       - A1: 平均融合替代门控
      no_multimodal - A2: 15 维单编码器，无分模态
      no_backbone   - A3: 去掉 TemporalBackbone
      no_med        - A4: 去掉 s(t)
      no_prox       - A5a: 去掉接近觉
      no_imu        - A5b: 去掉 IMU
    """

    def __init__(
        self,
        ablation: str = "full",
        d_model: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ablation = ablation.strip().lower()
        self.d_model = d_model

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Softplus(),
        )

        if self.ablation == "no_multimodal":
            self.flat_enc = ModalityEncoder(15, d_model, gru_hidden, dropout)
            self.backbone = TemporalBackbone(d_model, dropout)
            self.force_enc = None
            self.fusion = None
            return

        use_prox = self.ablation != "no_prox"
        use_imu = self.ablation != "no_imu"
        use_med = self.ablation != "no_med"
        self.use_prox = use_prox
        self.use_imu = use_imu
        self.use_med = use_med

        self.force_enc = ModalityEncoder(6, d_model, gru_hidden, dropout)
        self.prox_enc = ModalityEncoder(2, d_model, gru_hidden, dropout) if use_prox else None
        self.imu_enc = ModalityEncoder(6, d_model, gru_hidden, dropout) if use_imu else None
        self.med_enc = ModalityEncoder(1, d_model, gru_hidden, dropout) if use_med else None

        n_mod = 1 + int(use_prox) + int(use_imu) + int(use_med)
        if self.ablation == "no_gate":
            self.fusion = MeanFusion(d_model, n_mod=n_mod)
        else:
            self.fusion = GatedFusion(d_model, n_mod=n_mod)

        if self.ablation == "no_backbone":
            self.backbone = nn.Identity()
        else:
            self.backbone = TemporalBackbone(d_model, dropout)

    @staticmethod
    def split_modalities(x: torch.Tensor):
        force = torch.cat([x[..., 0:3], x[..., 7:10]], dim=-1)
        prox = torch.cat([x[..., 3:4], x[..., 10:11]], dim=-1)
        imu = torch.cat([x[..., 4:7], x[..., 11:14]], dim=-1)
        med = x[..., 14:15]
        return force, prox, imu, med

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        if self.ablation == "no_multimodal":
            h = self.backbone(self.flat_enc(x))
            y = self.head(h[:, -1, :])
            if return_gate:
                return y, None
            return y

        force, prox, imu, med = self.split_modalities(x)
        feats = [self.force_enc(force)]
        if self.use_prox:
            feats.append(self.prox_enc(prox))
        if self.use_imu:
            feats.append(self.imu_enc(imu))
        if self.use_med:
            feats.append(self.med_enc(med))

        fused, gate = self.fusion(feats)
        h = self.backbone(fused)
        y = self.head(h[:, -1, :])
        if return_gate:
            return y, gate
        return y


ABLATION_ALIASES = {
    "full": "full",
    "a0": "full",
    "cm_tfnet": "full",
    "no_gate": "no_gate",
    "a1": "no_gate",
    "mean_fusion": "no_gate",
    "no_multimodal": "no_multimodal",
    "a2": "no_multimodal",
    "flat": "no_multimodal",
    "no_backbone": "no_backbone",
    "a3": "no_backbone",
    "no_med": "no_med",
    "a4": "no_med",
    "no_s": "no_med",
    "no_prox": "no_prox",
    "a5a": "no_prox",
    "no_imu": "no_imu",
    "a5b": "no_imu",
}


def build_ablation(
    name: str,
    d_model: int = 64,
    gru_hidden: int = 64,
    dropout: float = 0.1,
) -> nn.Module:
    key = ABLATION_ALIASES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"unknown ablation: {name}")
    if key == "full":
        return CMTFNet(d_model, gru_hidden, dropout)
    return CMTFNetAblation(key, d_model, gru_hidden, dropout)
