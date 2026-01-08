from demucs.transformer import CrossTransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AFXCDT(nn.Module):
    """
    AFX-injected Cross Domain Transformer.
    元のCrossTransformerEncoderをラップし、エフェクトタイプによる条件付けを注入します。
    """
    def __init__(self, cdt_config, num_effects):
        super().__init__()
        self.num_effects = num_effects
        self.original_dim = cdt_config['dim']

        num_heads = cdt_config.get('num_heads', 8)
        self.cond_dim = num_heads * 4  # 論文と同程度の容量

        # one-hot → condition channel
        self.condition_projector = nn.Linear(num_effects, self.cond_dim)

        # Transformerは channel が増えた状態で動作
        conditioned_dim = self.original_dim + self.cond_dim
        assert conditioned_dim % num_heads == 0

        internal_cfg = cdt_config.copy()
        internal_cfg['dim'] = conditioned_dim
        self.internal_cdt = CrossTransformerEncoder(**internal_cfg)

        # 出力を元の次元に戻す
        self.out_proj_spec = nn.Conv1d(conditioned_dim, self.original_dim, 1)
        self.out_proj_time = nn.Conv1d(conditioned_dim, self.original_dim, 1)

    def forward(self, x, xt, effect_type_one_hot):
        """
        Args:
            x (torch.Tensor): 周波数ブランチの潜在表現 (B, C, F, T_spec)。
            xt (torch.Tensor): 時間ブランチの潜在表現 (B, C, T_time)。
            effect_type_one_hot (torch.Tensor): エフェクトタイプを示すone-hotベクトル (B, num_effects)。
        Returns:
            tuple: (conditioned_x, conditioned_xt)
        """
        # B: batch size, C: channels(dims), F: frequency bins, T_spec: sequence length
        B, C, F, T = x.shape

        # --- condition vector ---
        cond = self.condition_projector(effect_type_one_hot)  # (B, cond_dim)

        # --- freq branch ---
        cond_spec = cond[:, :, None, None].expand(B, self.cond_dim, F, T)
        x_cond = torch.cat([x, cond_spec], dim=1)  # (B, C+cond, F, T)

        # --- time branch ---
        _, _, Tt = xt.shape
        cond_time = cond[:, :, None].expand(B, self.cond_dim, Tt)
        xt_cond = torch.cat([xt, cond_time], dim=1)  # (B, C+cond, T)

        # --- transformer ---
        out_x, out_xt = self.internal_cdt(x_cond, xt_cond)

        # --- project back ---
        out_x = self.out_proj_spec(
            rearrange(out_x, 'b c f t -> b c (f t)')
        )
        out_x = rearrange(out_x, 'b c (f t) -> b c f t', f=F)

        out_xt = self.out_proj_time(out_xt)

        return out_x, out_xt
