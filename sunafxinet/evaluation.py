import torch
import numpy as np
import librosa
import soundfile as sf
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools

# モデルと推論関数のインポート
from sunafxinet import SunAFXiNet
from inference import iterative_inference
from constant import PARAM_DIMS, PARAM_RANGES, HDEMUCS_CONFIG, NUM_EFFECTS, EFFECT_MAP, EFFECT_PARAM_NAMES, PARAM_RANGES

# 評価指標ライブラリ
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
import auraloss

# エフェクトを再適用するためのPedalboard
import pedalboard

# ============================================================================
# 1. 評価用データセットクラス
# ============================================================================
class EvaluationDataset(Dataset):
    """
    評価用のデータセット。モードに応じて異なるデータペアを生成する。
    """
    def __init__(self, metadata_dir, mode='dry_recovery'):
        self.metadata_files = [os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.endswith('_metadata.json')]
        self.mode = mode
        self.data_pairs = self._create_pairs()
        if not self.data_pairs:
            raise FileNotFoundError(f"No valid data pairs found in {metadata_dir} for mode '{mode}'")

    def _create_pairs(self):
        pairs = []
        for meta_path in self.metadata_files:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            if self.mode == 'dry_recovery' or self.mode == 'wet_reproduction':
                # Part 2 & 3: (最終ウェット信号, 元のドライ信号) のペア
                pairs.append({
                    'wet_path': metadata['final_wet_signal_path'],
                    'dry_path': metadata['dry_signal_path'],
                    'metadata': metadata
                })
            elif self.mode == 'bypassed_signal':
                # Part 1: (中間ウェット信号, 1つ前の信号) のペア
                chain = metadata['effect_chain']
                for i in range(len(chain)):
                    input_path = metadata['intermediate_signals'][f'wet_{i}']
                    target_path = metadata['intermediate_signals'].get(f'wet_{i-1}', metadata['dry_signal_path'])
                    pairs.append({
                        'wet_path': input_path,
                        'dry_path': target_path, # このモードでは 'dry_path' は bypassed_signal を指す
                        'metadata': {'effect_chain': [chain[i]]} # 最後のAFX情報のみ
                    })
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

# ============================================================================
# 2. 評価パート別実行関数
# ============================================================================

### Part 1: SunAFXiNet自体の評価 ###
def evaluate_sunafxinet_step(model, data_loader, effect_map, device, mr_stft_loss):
    model.eval()
    si_snr_scores = []
    mr_stft_scores = []
    all_gt_types, all_pred_types = [], []
    param_mses = {name: [] for name in effect_map.keys()}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Part 1: Evaluating SunAFXiNet Step"):
            wet_path = batch['wet_path'][0]
            target_path = batch['dry_path'][0] # bypassed signal
            # 1. effect_chainリストから、最初の要素（エフェクト辞書）を取得
            gt_effect = batch['metadata']['effect_chain'][0]
            
            # 2. エフェクト辞書内の'type'キーの値（リスト）から、最初の要素（文字列）を取得
            gt_type_name = gt_effect['type'][0]
            
            # 3. パラメータ辞書を取得
            gt_params_dict = gt_effect['params']
            gt_type_idx = effect_map[gt_type_name]

            wet_signal, sr = librosa.load(wet_path, sr=48000, mono=True)
            target_signal, _ = librosa.load(target_path, sr=48000, mono=True)

            wet_tensor = torch.from_numpy(wet_signal).unsqueeze(0).unsqueeze(0).to(device)
            
            # 1ステップ推論
            s_hat_original, type_logits, param_predictions = model(wet_tensor)
            # --- 評価指標の計算 ---
            
            min_len = min(len(target_signal), s_hat_original.shape[-1])
            # 1. SI-SNRのためのテンソル準備 (両方とも1D: [L])
            # s_hat_original から一時的な変数を作成
            pred_for_sisnr = s_hat_original.squeeze()[:min_len]
            target_for_sisnr = torch.from_numpy(target_signal[:min_len]).to(device)
            si_snr = ScaleInvariantSignalNoiseRatio().to(device)(pred_for_sisnr, target_for_sisnr).item()
            si_snr_scores.append(si_snr)

            # 2. MR-STFTのためのテンソル準備 (両方とも3D: [B, C, L])
            # 再び s_hat_original から一時的な変数を作成
            # [1, 1, 1, L] -> [1, 1, L]
            pred_for_mrstft = s_hat_original.squeeze(0)[:, :, :min_len]
            # [L] -> [1, L] -> [1, 1, L]
            target_for_mrstft = target_for_sisnr.unsqueeze(0).unsqueeze(0)
            print(pred_for_mrstft.shape, target_for_mrstft.shape)
            mr_stft = mr_stft_loss(pred_for_mrstft, target_for_mrstft).item()
            mr_stft_scores.append(mr_stft)

            pred_type_idx = torch.argmax(type_logits, dim=1).item()
            all_gt_types.append(gt_type_idx)
            all_pred_types.append(pred_type_idx)

            if pred_type_idx == gt_type_idx:
                pred_params_raw = param_predictions[gt_type_name].squeeze().cpu().numpy()
                pred_params_normalized = np.atleast_1d(pred_params_raw)
                # 1. 逆正規化のためのヘルパー関数を定義
                def de_normalize_param(value, min_val, max_val):
                    return value * (max_val - min_val) + min_val

                # 2. 予測された正規化値を、元のスケールに逆正規化する
                de_normalized_pred_params = []
                param_names = list(PARAM_RANGES[gt_type_name].keys())
                for i, param_name in enumerate(param_names):
                    min_val, max_val = PARAM_RANGES[gt_type_name][param_name]
                    real_value = de_normalize_param(pred_params_normalized[i], min_val, max_val)
                    de_normalized_pred_params.append(real_value)

                # 3. 正解パラメータ（本来の値）を取得
                gt_params = [v.item() for v in gt_params_dict.values()]
                
                # 4. 本来の値同士でMSEを計算する
                mse = mean_squared_error(gt_params, de_normalized_pred_params[:len(gt_params)])
                param_mses[gt_type_name].append(mse)

    # --- 結果の集計 ---
    accuracy = accuracy_score(all_gt_types, all_pred_types)
    avg_param_mse = {k: np.mean(v) if v else 0 for k, v in param_mses.items()}
    return {
        'avg_si_snr': np.mean(si_snr_scores), 
        'avg_mr_stft': np.mean(mr_stft_scores), 
        'afx_accuracy': accuracy, 
        'param_mse': avg_param_mse
    }

### Part 2: ドライ信号復元評価 table4 ###
def evaluate_dry_signal_recovery(model, data_loader, effect_map, device, mr_stft_loss):
    model.eval()
    si_snr_scores, mr_stft_scores = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Part 2: Evaluating Dry Signal Recovery"):
            wet_path = batch['wet_path'][0]
            true_dry_path = batch['dry_path'][0]
            
            wet_signal, sr = librosa.load(wet_path, sr=48000, mono=True)
            true_dry_signal, _ = librosa.load(true_dry_path, sr=48000, mono=True)
            
            _, pred_dry_signal = iterative_inference(model, wet_signal, effect_map, device)

            min_len = min(len(true_dry_signal), len(pred_dry_signal))
            # 1. SI-SNRのためのテンソル準備 (1D: [L])
            pred_for_sisnr = torch.from_numpy(pred_dry_signal[:min_len]).to(device)
            target_for_sisnr = torch.from_numpy(true_dry_signal[:min_len]).to(device)
            si_snr = ScaleInvariantSignalNoiseRatio().to(device)(pred_for_sisnr, target_for_sisnr).item()
            si_snr_scores.append(si_snr)

            # 2. MR-STFTのためのテンソル準備 (3D: [B, C, L])
            # 1Dテンソルから .unsqueeze() を2回使って3Dテンソルを作成
            pred_for_mrstft = pred_for_sisnr.unsqueeze(0).unsqueeze(0)
            target_for_mrstft = target_for_sisnr.unsqueeze(0).unsqueeze(0)
            mr_stft = mr_stft_loss(pred_for_mrstft, target_for_mrstft).item()
            mr_stft_scores.append(mr_stft)

    return {'avg_si_snr': np.mean(si_snr_scores), 'avg_mr_stft': np.mean(mr_stft_scores)}

### Part 3: ウェット信号再現評価 ###
def evaluate_wet_signal_reproduction(model, data_loader, effect_map, device, mr_stft_loss):
    model.eval()
    si_sdr_i_x_hat, mr_stft_i_x_hat = [], []
    si_sdr_i_x, mr_stft_i_x = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Part 3: Evaluating Wet Signal Reproduction"):
            wet_path = batch['wet_path'][0]
            true_dry_path = batch['dry_path'][0]

            wet_signal, sr = librosa.load(wet_path, sr=48000, mono=True)
            true_dry_signal, _ = librosa.load(true_dry_path, sr=48000, mono=True)

            estimated_chain, pred_dry_signal = iterative_inference(model, wet_signal, effect_map, device)

            # 不要なヘルパー関数を削除し、try...exceptブロックのみを残す
            try:
                # `iterative_inference`から返された辞書 `fx['params']` をそのまま展開する
                board = pedalboard.Pedalboard([
                    getattr(pedalboard, fx['type'])(**fx['params'])
                    for fx in estimated_chain
                ])
            except Exception as e:
                print(f"Error creating pedalboard for {os.path.basename(wet_path)}, skipping sample: {e}")
                continue # エラーが発生した場合はこのサンプルをスキップ
            
            # 1. 推定されたドライ信号 x_hat からウェット信号を再現
            reproduced_wet_from_x_hat = board(pred_dry_signal, sr) if estimated_chain else pred_dry_signal
            
            # 2. 正解のドライ信号 x からウェット信号を再現
            reproduced_wet_from_x = board(true_dry_signal, sr) if estimated_chain else true_dry_signal

            # --- 評価指標の計算 ---
            min_len_y = len(wet_signal)
            wet_tensor = torch.from_numpy(wet_signal).to(device)
            
            # SI-SDRiとMR-STFTiの計算
            def get_metrics_i(base_signal, reproduced_signal):
                min_len = min(len(base_signal), len(reproduced_signal), min_len_y)
                base_t = torch.from_numpy(base_signal[:min_len]).to(device)
                repro_t = torch.from_numpy(reproduced_signal[:min_len]).to(device)
                
                # SI-SDRの計算 (1Dテンソル)
                si_sdr_y_base = ScaleInvariantSignalNoiseRatio().to(device)(base_t, wet_tensor[:min_len]).item()
                si_sdr_y_repro = ScaleInvariantSignalNoiseRatio().to(device)(repro_t, wet_tensor[:min_len]).item()
                si_sdr_i = si_sdr_y_repro - si_sdr_y_base

                # MR-STFTの計算 (3Dテンソル)
                base_t_3d = base_t.unsqueeze(0).unsqueeze(0)
                wet_tensor_3d = wet_tensor[:min_len].unsqueeze(0).unsqueeze(0)
                repro_t_3d = repro_t.unsqueeze(0).unsqueeze(0)
                mrstft_y_base = mr_stft_loss(base_t_3d, wet_tensor_3d).item()
                mrstft_y_repro = mr_stft_loss(repro_t_3d, wet_tensor_3d).item()
                mr_stft_i = mrstft_y_base - mrstft_y_repro
                
                return si_sdr_i, mr_stft_i

            sdr_i_xh, stft_i_xh = get_metrics_i(pred_dry_signal, reproduced_wet_from_x_hat)
            si_sdr_i_x_hat.append(sdr_i_xh)
            mr_stft_i_x_hat.append(stft_i_xh)

            sdr_i_x, stft_i_x = get_metrics_i(true_dry_signal, reproduced_wet_from_x)
            si_sdr_i_x.append(sdr_i_x)
            mr_stft_i_x.append(stft_i_x)
            
    return {
        'avg_si_sdr_i_x_hat': np.mean(si_sdr_i_x_hat), 'avg_mr_stft_i_x_hat': np.mean(mr_stft_i_x_hat),
        'avg_si_sdr_i_x': np.mean(si_sdr_i_x), 'avg_mr_stft_i_x': np.mean(mr_stft_i_x)
    }

# ============================================================================
# 4. メイン実行ブロック
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SunAFXiNet Full Evaluation Script")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--test-data-dir', type=str, required=True, help='Path to the directory with test set metadata.')
    parser.add_argument('--eval-mode', type=str, required=True, choices=['bypassed_signal', 'dry_recovery', 'wet_reproduction'], help='Which part of the evaluation to run.')
    parser.add_argument('--results-file', type=str, default='evaluation_results.json', help='Path to save the evaluation results JSON file.')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Model Configuration (Must match training) ---    
    hdemucs_config = HDEMUCS_CONFIG
    hdemucs_config['samplerate'] = 48000
    
    print(f"Loading model from {args.model_path}...")
    model = SunAFXiNet(hdemucs_config, NUM_EFFECTS, PARAM_DIMS)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded successfully.")

    print(f"Loading test data from {args.test_data_dir} for mode '{args.eval_mode}'...")
    eval_dataset = EvaluationDataset(metadata_dir=args.test_data_dir, mode=args.eval_mode)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Found {len(eval_dataset)} evaluation samples.")

    # MR-STFT Loss (共通で使用)
    mr_stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096], 
        hop_sizes=[256, 512, 1024], 
        win_lengths=[1024, 2048, 4096],
        sample_rate=48000
    ).to(DEVICE)

    # --- Run Selected Evaluation ---
    if args.eval_mode == 'bypassed_signal':
        results = evaluate_sunafxinet_step(model, eval_loader, EFFECT_MAP, DEVICE, mr_stft_loss)
    elif args.eval_mode == 'dry_recovery':
        results = evaluate_dry_signal_recovery(model, eval_loader, EFFECT_MAP, DEVICE, mr_stft_loss)
    elif args.eval_mode == 'wet_reproduction':
        results = evaluate_wet_signal_reproduction(model, eval_loader, EFFECT_MAP, DEVICE, mr_stft_loss)
    else:
        raise ValueError("Invalid evaluation mode selected.")

    # --- Display and Save Results ---
    print("\n" + "="*50)
    print(f"      Evaluation Results for mode: {args.eval_mode}")
    print("="*50)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    print("="*50)

    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {args.results_file}")

if __name__ == '__main__':
    main()
