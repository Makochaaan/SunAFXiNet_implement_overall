import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HAfx(nn.Module):
    """
    AFX config estimator (hafx).
    HTDemucsのボトルネックから得られる周波数領域の潜在表現を入力とし、
    エフェクトのタイプとパラメータを推定します。
    """
    def __init__(self, input_dim, num_effects, param_dims, hidden_dim=512):
        super().__init__()
        self.num_effects = num_effects
        self.param_dims = param_dims

        # 潜在表現をフラット化して1D畳み込みで処理
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # タイプ推定用の全結合層
        self.type_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # meanとmaxを連結
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_effects),
        )

        # パラメータ推定用の全結合層（各エフェクトタイプごとにヘッドを持つ）
        self.param_fcs = nn.ModuleDict()
        for effect_name, p_dim in param_dims.items():
            self.param_fcs[effect_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(hidden_dim, p_dim)
            )

    def forward(self, z_spec):
        """
        Args:
            z_spec (torch.Tensor): 周波数ブランチの潜在表現 (B, C, F, T)。
        Returns:
            tuple: (type_logits, param_predictions)
        """
        # (B, C, F, T) -> (B, C, F*T)
        z_flat = rearrange(z_spec, 'b c f t -> b c (f t)')

        h = self.conv_blocks(z_flat)
        # Global Pooling
        h_mean = torch.mean(h, dim=2)
        h_max, _ = torch.max(h, dim=2)
        pooled_h = h_mean + h_max

        # タイプ推定
        type_logits = self.type_fc(pooled_h)

        # パラメータ推定
        param_predictions = {}
        for effect_name, fc_head in self.param_fcs.items():
            param_predictions[effect_name] = fc_head(pooled_h)

        return type_logits, param_predictions
