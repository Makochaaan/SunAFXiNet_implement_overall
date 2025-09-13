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
        
        self.condition_proj_dim = num_heads * 4  # e.g., 8 * 4 = 32
        self.condition_projector = nn.Linear(num_effects, self.condition_proj_dim)

        # 条件ベクトルを連結するため、Transformerの入力次元が増加
        conditioned_dim = self.original_dim + self.condition_proj_dim
        assert conditioned_dim % num_heads == 0

        # 内部に持つCDTのための設定を作成
        internal_cdt_config = cdt_config.copy()
        internal_cdt_config['dim'] = conditioned_dim
        self.internal_cdt = CrossTransformerEncoder(**internal_cdt_config)

        # 出力次元を元に戻すためのプロジェクション層
        self.output_proj_spec = nn.Conv1d(conditioned_dim, self.original_dim, 1)
        self.output_proj_time = nn.Conv1d(conditioned_dim, self.original_dim, 1)
        # print("AFXCDT initialized with conditioned_dim:", conditioned_dim)

    def forward(self, x, xt, effect_type_one_hot):
        """
        Args:
            x (torch.Tensor): 周波数ブランチの潜在表現 (B, C, F, T_spec)。
            xt (torch.Tensor): 時間ブランチの潜在表現 (B, C, T_time)。
            effect_type_one_hot (torch.Tensor): エフェクトタイプを示すone-hotベクトル (B, num_effects)。
        Returns:
            tuple: (conditioned_x, conditioned_xt)
        """
        # print("start AFXCDT conditioning")
        # B: batch size, C: channels(dims), F: frequency bins, T_spec: sequence length
        # --- 周波数ブランチの条件付け ---
        B, C, F, T_spec = x.shape
        # condition_spec = effect_type_one_hot.view(B, self.num_effects, 1, 1).expand(B, self.num_effects, F, T_spec)
        # conditioned_x = torch.cat([x, condition_spec], dim=1)
        # 1. 【翻訳】one-hotベクトルを射影層に通して、豊かな特徴量に変換
        # 入力: effect_type_one_hot (形状: [B, 5])
        # 出力: condition_vec (形状: [B, 32])
        condition_vec = self.condition_projector(effect_type_one_hot)

        # 2. 音声データと条件ベクトルの形状を整える
        rearranged_x = rearrange(x, 'b c f t -> b (f t) c')
        # 3. 【重要】射影後のcondition_vecを拡張・整形する
        # (B, 32) -> (B, 1, 32) -> (B, L, 32)
        rearranged_effect_type = condition_vec.unsqueeze(1).expand(-1, rearranged_x.shape[1], -1)

        # 4. 2つのテンソルを連結する
        # (B, L, 768) と (B, L, 32) を連結 -> (B, L, 800)
        conditioned_x = torch.cat([rearranged_x, rearranged_effect_type], dim=2)

        conditioned_x = rearrange(conditioned_x, 'b c (f t) -> b t f c', f=x.shape[2])


        # --- 時間ブランチの条件付け ---
        _B, _C, T_time = xt.shape
        # condition_time = effect_type_one_hot.view(B, self.num_effects, 1).expand(B, self.num_effects, T_time)
        # conditioned_xt = torch.cat([xt, condition_time], dim=1)
        condition_time = self.condition_projector(effect_type_one_hot)
        rearranged_xt = rearrange(xt, 'b c t -> b (t) c')
        rearranged_effect_time = condition_time.unsqueeze(1).expand(-1, rearranged_xt.shape[1], -1)
        conditioned_xt = torch.cat([rearranged_xt, rearranged_effect_time], dim=2)
        conditioned_xt = rearrange(conditioned_xt, 'b c t -> b t c')

        # print("conditioned_x shape:", conditioned_x.shape)
        # print("conditioned_xt shape:", conditioned_xt.shape)

        # --- 内部Transformerによる処理 ---
        # 元のHTDemucsと同様に、bottom_channelsの処理は外部で行われると仮定
        out_x, out_xt = self.internal_cdt(conditioned_x, conditioned_xt)

        # --- 出力プロジェクション ---
        # Transformerの出力は(B, C, F, T)と(B, C, T)なので、1D Convでチャネル数を調整
        out_x_flat = rearrange(out_x, 'b c f t -> b c (f t)')
        proj_x_flat = self.output_proj_spec(out_x_flat)
        proj_x = rearrange(proj_x_flat, 'b c (f t) -> b c f t', f=F)

        proj_xt = self.output_proj_time(out_xt)

        return proj_x, proj_xt
