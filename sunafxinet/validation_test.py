import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

# 必要なクラスと関数をインポート
from sunafxinet import SunAFXiNet
from evaluation import EvaluationDataset # 既存の評価用Datasetを再利用
from constant import PARAM_DIMS, HDEMUCS_CONFIG, NUM_EFFECTS, EFFECT_MAP

# ============================================================================
# 1. ΔMAEを計算するメイン関数
# ============================================================================

def calculate_delta_mae_distributions(model, data_loader, effect_map, device):
    """
    検証データセット全体でΔMAEの分布を計算する。
    """
    model.eval()
    
    results = []
    inv_effect_map = {v: k for k, v in effect_map.items()}
    num_effects = len(effect_map)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating ΔMAE Distributions"):
            print(batch)
            wet_path = batch['wet_path'][0]
            metadata = batch['metadata']
            
            ground_truth_chain = [fx['type'][0] for fx in metadata['effect_chain']]
            
            wet_signal, sr = librosa.load(wet_path, sr=48000, mono=True)
            
            current_signal_np = wet_signal
            predicted_effects_indices = []

            # 論文に従い、4回（全エフェクト数分）の反復を行う
            for i in range(num_effects):
                input_signal_tensor = torch.from_numpy(current_signal_np).unsqueeze(0).unsqueeze(0).to(device)
                
                # 1ステップ推論
                s_hat_tensor, type_logits, _ = model(input_signal_tensor)
                
                # --- AFXタイプマスキング ---
                # すでに推定されたエフェクトの確率を非常に低くして、再度選択されないようにする
                for pred_idx in predicted_effects_indices:
                    type_logits[0, pred_idx] = -torch.inf
                
                pred_type_idx = torch.argmax(type_logits, dim=1).item()
                predicted_effects_indices.append(pred_type_idx)
                pred_type_name = inv_effect_map[pred_type_idx]
                
                # --- ΔMAEの計算 ---
                # LMAE(sˆ, u)
                # s_hat_tensorの形状: [1,1,1,L], input_signal_tensorの形状: [1,1,L]
                # squeeze()で次元を合わせる
                l1_loss = F.l1_loss(s_hat_tensor.squeeze(), input_signal_tensor.squeeze())
                
                # ||u||1
                l1_norm = torch.linalg.norm(input_signal_tensor.squeeze(), ord=1)
                
                delta_mae = (l1_loss / l1_norm).item()
                
                # --- IN/NOT IN target chain の判定 ---
                status = "IN target chain" if pred_type_name in ground_truth_chain else "NOT IN target chain"
                
                results.append({
                    'delta_mae': delta_mae,
                    'predicted_effect': pred_type_name,
                    'status': status,
                    'iteration': i + 1
                })
                
                # 次のイテレーションのために信号を更新
                current_signal_np = s_hat_tensor.squeeze().cpu().numpy()

    return pd.DataFrame(results)

# ============================================================================
# 2. 結果を可視化・分析する関数
# ============================================================================

def visualize_and_analyze(df, output_dir):
    """
    計算結果をバイオリンプロットで可視化し、閾値を計算する。
    """
    print("\nVisualizing ΔMAE distributions...")
    
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df, x='predicted_effect', y='delta_mae', hue='status', split=True, inner='quartile', palette={'IN target chain': 'skyblue', 'NOT IN target chain': 'coral'})
    
    plt.title('ΔMAE Distributions for each AFX Type')
    plt.ylabel('ΔMAE (Normalized L1 Difference)')
    plt.xlabel('Predicted AFX Type')
    plt.yscale('log') # 論文の図5は対数スケールのように見えるため適用
    plt.grid(True, which="both", ls="--")
    
    plot_path = os.path.join(output_dir, "delta_mae_violin_plot.png")
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.close()

    # --- 閾値の計算 ---
    print("\nCalculating stopping thresholds (Top 20 percentile of 'NOT IN' distribution)...")
    
    not_in_df = df[df['status'] == 'NOT IN target chain']
    
    thresholds = {}
    for effect_type in not_in_df['predicted_effect'].unique():
        # 上位20パーセンタイルは、80パーセンタイル値と同じ
        threshold = not_in_df[not_in_df['predicted_effect'] == effect_type]['delta_mae'].quantile(0.80)
        thresholds[effect_type] = threshold
        print(f"  - Threshold for {effect_type}: {threshold:.6f}")
        
    thresholds_path = os.path.join(output_dir, "stopping_thresholds.json")
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=4)
    print(f"Thresholds saved to {thresholds_path}")
    
    return thresholds

# ============================================================================
# 3. メイン実行ブロック
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SunAFXiNet ΔMAE Distribution Calculation Script")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.pth) file.')
    # parser.add_argument('--valid-data-dir', type=str, required=True, help='Path to the directory with validation set metadata.')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Directory to save results (CSV, plot, thresholds).')
    args = parser.parse_args()

    VALID_DIR = '../../../dataset/sunafxinet/wet_signal_2way/valid'

    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    hdemucs_config = HDEMUCS_CONFIG
    hdemucs_config['samplerate'] = 48000
    
    print(f"Loading model from {args.model_path}...")
    model = SunAFXiNet(hdemucs_config, NUM_EFFECTS, PARAM_DIMS)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    print("Model loaded successfully.")

    print(f"Loading validation data from {VALID_DIR}...")
    # 'dry_recovery'モードは(最終ウェット, 元のドライ)のペアと完全なメタデータを返すため、この分析に適している
    valid_dataset = EvaluationDataset(metadata_dir=VALID_DIR, mode='dry_recovery')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Found {len(valid_dataset)} validation samples.")

    # --- Run Calculation ---
    results_df = calculate_delta_mae_distributions(model, valid_loader, EFFECT_MAP, DEVICE)
    
    # Save raw data
    csv_path = os.path.join(args.output_dir, "delta_mae_raw_data.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nRaw ΔMAE data saved to {csv_path}")

    # --- Visualize and Analyze ---
    visualize_and_analyze(results_df, args.output_dir)

if __name__ == '__main__':
    main()

# ```

# ### このスクリプトの使い方

# 1.  **必要なライブラリのインストール**:
#     可視化のために`seaborn`と`matplotlib`が必要です。
#     ```bash
#     pip install pandas seaborn matplotlib
#     ```

# 2.  **ファイルの準備**:
#     * このコードを `calculate_delta_mae.py` という名前で保存します。
#     * 学習済みの最終モデル (`sunafxinet_final.pth`) を用意します。
#     * 分割済みの**検証用（validation）データセット**のメタデータが格納されたディレクトリ（例: `./wet_signal/valid`）が必要です。

# 3.  **スクリプトの実行**:
#     ターミナルで以下のコマンドを実行します。
#     ```bash
#     python calculate_delta_mae.py --model-path ./sunafxinet_final.pth --valid-data-dir ./wet_signal/valid --output-dir ./delta_mae_analysis
    
