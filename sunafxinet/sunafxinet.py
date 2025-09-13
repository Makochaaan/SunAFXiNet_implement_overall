import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from demucs.htdemucs import HTDemucs
from afxcdt import AFXCDT
from hafx import HAfx

class SunAFXiNet(HTDemucs):
    """
    HTDemucsを継承し、AFX除去とパラメータ推定のために改造した完全なモデル。
    """
    def __init__(self, hdemucs_config, num_effects, param_dims):
        # 信号復元タスクなので、出力ソースは1つに設定
        hdemucs_config['sources'] = ['restored']
        
        # HTDemucsの初期化
        super().__init__(**hdemucs_config)
        
        # Transformerの次元数を取得
        # 【修正点】self.growthやself.depthはHTDemucsの__init__で属性として保存されないため、
        # コンフィグ辞書から直接値を取得する
        growth = hdemucs_config.get('growth', 2) # デフォルト値は元のコードに合わせる
        depth = hdemucs_config.get('depth', 4)   # デフォルト値は元のコードに合わせる
        transformer_channels = self.channels * (growth ** (depth - 1))
        
        if self.bottom_channels:
            transformer_channels = self.bottom_channels
            
        # HAfxモジュールの初期化
        # print(f"✅ Initializing transformer with dimension (d_model): {transformer_channels}")
        self.hafx = HAfx(
            input_dim=transformer_channels, 
            num_effects=num_effects, 
            param_dims=param_dims
        )
        
        # AFXCDTモジュールの初期化
        # print("start AFXCDT initialization")
        if self.crosstransformer is not None:
            # CrossTransformerEncoderに設定を取得するメソッドがないと仮定し、
            # __init__の引数から手動で設定を再構築
            cdt_config = {k.replace('t_', ''): v for k, v in hdemucs_config.items() if k.startswith('t_')}
            # print(f"config[dim] = {transformer_channels}")
            cdt_config['dim'] = transformer_channels
            if 'layers' in cdt_config: cdt_config['num_layers'] = cdt_config.pop('layers')
            if 'heads' in cdt_config: cdt_config['num_heads'] = cdt_config.pop('heads')
            # print(cdt_config)
            self.afx_cdt = AFXCDT(cdt_config, num_effects)
            # 【修正】元のcrosstransformerを無効化し、意図しない呼び出しを防ぐ
            self.crosstransformer = None
        else:
            self.afx_cdt = None

    def forward(self, mix, afx_type_condition=None):
        """
        HTDemucs.forwardをオーバーライドし、AFX推定と条件付けを組み込む。

        Args:
            mix (torch.Tensor): 入力ウェット信号 (B, C, L)。
            afx_type_condition (torch.Tensor, optional):
                学習の第1段階で使用するグラウンドトゥルースのone-hotベクトル (B, num_effects)。
                Noneの場合はhafxの推定値を使用する。
        Returns:
            tuple: (s_hat, type_logits, param_predictions)
                s_hat (torch.Tensor): 復元された信号 (B, 1, C, L)。
                type_logits (torch.Tensor): (B, num_effects)
                param_predictions (dict): {'Distortion': (B, p_dim),...}
        """
        # ===== HTDemucs.forward() からのコード（エンコーダ部分）=====
        # print("start SunAFXiNet Encoder")
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
          if self.training:
            self.segment = length / self.samplerate
          else:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag
        B, C, Fq, T = x.shape
        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # ===== ここからがSunAFXiNetの介入部分 =====

        # 1. hafxでAFXを推定 (Transformerに入る直前の周波数表現 `x` を使用)
        # print("start HAfx inference")
        type_logits, param_predictions = self.hafx(x)

        # 2. 条件付け用のone-hotベクトルを決定
        # print("decide one-hot vector")
        if afx_type_condition is not None:
            # 学習ステージ1: グラウンドトゥルースで条件付け
            condition_one_hot = afx_type_condition
        else:
            # 学習ステージ2 & 推論: hafxの推定値で条件付け
            pred_indices = torch.argmax(type_logits, dim=1)
            condition_one_hot = F.one_hot(pred_indices, num_classes=self.hafx.num_effects).float()

        # 3. AFXCDT (crosstransformer) を呼び出し
        # print("start AFXCDT inference")
        if self.afx_cdt is not None: # self.crosstransformerはAFXCDTインスタンス
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(self.channel_upsampler(rearrange(x, "b c f t -> b c (f t)")), "b c (f t) -> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)
            
            # AFXCDTに条件ベクトルを追加で渡す
            # print(f"🚨 Shape of 'x' before afx_cdt: {x.shape}")
            # print(f"🚨 Shape of 'xt' before afx_cdt: {xt.shape}")
            x, xt = self.afx_cdt(x, xt, effect_type_one_hot=condition_one_hot)
            
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(self.channel_downsampler(rearrange(x, "b c f t -> b c (f t)")), "b c (f t) -> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        # ===== HTDemucs.forward() からのコード（デコーダ部分）=====
        # print("start SunAFXiNet Decoder")
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    xt, _ = tdec(pre[:, :, 0], None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)
        
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T) * std[:, None] + mean[:, None]
        zout = self._mask(z, x)
        s_hat_spec = self._ispec(zout, length)
        xt = (xt.view(B, S, -1, length) * stdt[:, None] + meant[:, None])
        s_hat = xt + s_hat_spec

        if length_pre_pad:
            s_hat = s_hat[..., :length_pre_pad]

        return s_hat, type_logits, param_predictions
