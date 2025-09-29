import torch
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import auraloss # auralossのインストールが必要です: pip install auraloss
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import json
import glob
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from sunafxinet import SunAFXiNet
from constant import EFFECT_TYPES, PARAM_DIMS, NUM_EFFECTS, EFFECT_MAP, INV_EFFECT_MAP, HDEMUCS_CONFIG, BATCH_SIZE, LR_STAGE1, LR_STAGE2, EPOCHS_STAGE1, EPOCHS_STAGE2, LAMBDA_STFT, SAMPLE_RATE, PARAM_RANGES, DATASET_DIR

def plot_losses(train_losses, valid_losses, stage_name, save_path=None):
    """訓練損失と検証損失の折れ線グラフを描画する"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, 'o-', label='Validation Loss', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{stage_name} Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

class AFXChainDataset(Dataset):
    def __init__(self, metadata_dir, effect_map, param_dims):
        """
        SunAFXiNet学習用のカスタムデータセット。

        Args:
            metadata_dir (str): 生成されたメタデータJSONファイルが格納されているディレクトリ。
            effect_map (dict): エフェクト名を整数インデックスにマッピングする辞書。
        """
        self.metadata_files = glob.glob(os.path.join(metadata_dir, '*_metadata.json'))
        self.effect_map = effect_map
        self.training_pairs = self._create_training_pairs()
        self.param_dims = param_dims
        self.max_param_dim = max(param_dims.values()) if param_dims else 0
        self.training_pairs = self._create_training_pairs()

    def _create_training_pairs(self):
        pairs = []
        for meta_file in self.metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            chain = metadata['effect_chain']
            num_effects = len(chain)

            for i in range(num_effects):
                # 入力信号 u(k)
                input_signal_path = metadata['intermediate_signals'][f'wet_{i}']

                # ターゲット信号 s(k)
                target_signal_path = metadata['intermediate_signals'].get(f'wet_{i-1}', metadata['dry_signal_path'])

                # 最後のAFXの情報
                last_effect = chain[i]

                pairs.append({
                    'input_path': input_signal_path,
                    'target_path': target_signal_path,
                    'effect_type': last_effect['type'],
                    'effect_params': last_effect['params']
                })
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        pair = self.training_pairs[idx]

        # 音声ファイルの読み込み
        input_audio, _ = sf.read(pair['input_path'], dtype='float32')
        target_audio, _ = sf.read(pair['target_path'], dtype='float32')

        # チャンネル次元を追加 (B, C, L) -> (1, L)
        input_audio = torch.from_numpy(input_audio).unsqueeze(0)
        target_audio = torch.from_numpy(target_audio).unsqueeze(0)

        # エフェクトタイプのラベル
        effect_type_label = self.effect_map[pair['effect_type']]

        # パラメータの正規化
        params_dict = pair['effect_params']
        
        # 各エフェクトタイプのパラメータ範囲を定義
        param_ranges = PARAM_RANGES
        # パラメータを正規化してベクトルに変換
        normalized_params = {}
        effect_type = pair['effect_type']
        
        # 各パラメータを[0,1]の範囲に正規化
        for param_name, value in params_dict.items():
            if effect_type in param_ranges and param_name in param_ranges[effect_type]:
                min_val, max_val = param_ranges[effect_type][param_name]
                normalized_params[param_name] = (value - min_val) / (max_val - min_val)
            else:
                normalized_params[param_name] = value  # 範囲が定義されていない場合はそのまま
        
        # 正規化されたパラメータを固定長ベクトルに変換
        param_vector = self._vectorize_params(effect_type, normalized_params)

        return {
            'input_audio': input_audio,
            'target_audio': target_audio,
            'effect_type_label': torch.tensor(effect_type_label, dtype=torch.long),
            'param_vector': torch.tensor(param_vector, dtype=torch.float32)
        }

    def _vectorize_params(self, effect_type, params_dict):
        # この関数は、すべてのエフェクトのパラメータを統一されたベクトル形式に変換する
        # 例として、最大パラメータ数を持つReverbに合わせてベクトル長を4とする
        vector = np.zeros(self.max_param_dim, dtype=np.float32)
        
        param_values = []
        if effect_type == 'Distortion':
            param_values.append(params_dict['drive_db'])
        elif effect_type == 'Chorus':
            param_values.extend([params_dict['rate_hz'], params_dict['depth'], params_dict['mix']])
        elif effect_type == 'Delay':
            param_values.extend([params_dict['delay_seconds'], params_dict['feedback'], params_dict['mix']])
        elif effect_type == 'Reverb':
            param_values.extend([params_dict['room_size'], params_dict['damping'], params_dict['wet_level'], params_dict['dry_level']])
        
        # パディングされたベクトルに値を設定
        if param_values:
            vector[:len(param_values)] = param_values
            
        return vector

# --- 使用例 ---
# effect_map = {name: i for i, name in enumerate(EFFECT_TYPES)}
# dataset = AFXChainDataset(metadata_dir='/home/depontes25/Desktop/Research/Clone/SunAFXiNet/wet_signal/', effect_map=effect_map)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# --- 検証ループ関数 ---
def validate_epoch(model, valid_loader, criterion_mae, criterion_stft, criterion_ce, criterion_mse, device, stage):
    """1エポック分の検証を行い、平均損失を返す"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Validating [Stage {stage}]", leave=False):
            input_audio = batch['input_audio'].to(device)
            target_audio = batch['target_audio'].to(device)
            effect_type_label = batch['effect_type_label'].to(device)
            
            if stage == 1:
                condition_one_hot = F.one_hot(effect_type_label, num_classes=NUM_EFFECTS).float()
                s_hat, _, _ = model(input_audio, afx_type_condition=condition_one_hot)
                s_hat_squeezed = s_hat.squeeze(1)
                loss_mae = criterion_mae(s_hat_squeezed, target_audio)
                loss_stft = criterion_stft(s_hat_squeezed, target_audio)
                loss = loss_mae + LAMBDA_STFT * loss_stft
            
            elif stage == 2:
                param_vector = batch['param_vector'].to(device)
                _, type_logits, param_predictions = model(input_audio)
                
                loss_ce = criterion_ce(type_logits, effect_type_label)
                loss_mse = 0
                for type_idx, type_name in INV_EFFECT_MAP.items():
                    mask = (effect_type_label == type_idx)
                    if mask.any():
                        p_dim = PARAM_DIMS[type_name]
                        pred = param_predictions[type_name][mask, :p_dim]
                        true = param_vector[mask, :p_dim]
                        loss_mse += criterion_mse(pred, true)
                loss = loss_ce + loss_mse # lambda_paramは後で追加可能
            
            total_loss += loss.item()
            
    return total_loss / len(valid_loader)

# --- 推論関数 (完成版) ---
def iterative_inference(model, wet_signal, effect_map, device, max_iterations=5, sr=48000):
    """
    学習済みSunAFXiNetモデルを用いて反復的にエフェクトチェーンを推定・除去する。
    """
    model.eval()
    
    # デバイスの指定と、入力信号のテンソル化
    current_signal = torch.from_numpy(wet_signal).unsqueeze(0).unsqueeze(0).to(device)
    
    # 推定されたエフェクトチェーンを格納するリスト
    estimated_chain_reversed = []
    
    inv_effect_map = {v: k for k, v in effect_map.items()}
    no_effect_idx = len(effect_map) # "no effect"クラスは最後のインデックスと仮定

    with torch.no_grad():
        for i in range(max_iterations):
            s_hat_tensor, type_logits, param_predictions = model(current_signal)
            pred_type_idx = torch.argmax(type_logits, dim=1).item()

            if pred_type_idx == no_effect_idx:
                print("Stopping: 'no effect' class predicted.")
                break

            prev_energy = torch.mean(current_signal**2)
            new_energy = torch.mean(s_hat_tensor**2)
            if torch.abs(prev_energy - new_energy) / prev_energy < 1e-5:
                print("Stopping: Signal energy change is negligible.")
                break

            pred_type_name = inv_effect_map[pred_type_idx]
            pred_params = param_predictions[pred_type_name].squeeze(0).cpu().numpy()
            
            estimated_chain_reversed.append({
                'type': pred_type_name,
                'params': pred_params.tolist()
            })
            
            print(f"Iteration {i + 1}: Detected {pred_type_name}")
            current_signal = s_hat_tensor
            
    final_dry_signal = current_signal.squeeze(0).squeeze(0).cpu().numpy()
    estimated_chain = list(reversed(estimated_chain_reversed))

    return estimated_chain, final_dry_signal

# --- 学習プログラム ---

def train_stage1(model, train_loader, valid_loader, optimizer, criterion_mae, criterion_stft, lambda_stft, device, epochs):
    """第1段階: バイパス信号推定器 (hsig) の学習"""
    model.train()
    # hafxを凍結
    for param in model.hafx.parameters():
        param.requires_grad = False
    # hsig (hafx以外の部分) を学習可能に
    for name, param in model.named_parameters():
        if 'hafx' not in name:
            param.requires_grad = True

    print("--- Starting Training Stage 1: Training Bypassed Signal Estimator (hsig) ---")
    train_losses = []  # loss値を記録するリスト
    valid_losses = []
    best_valid_loss = float('inf')
    equalization_count = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_audio = batch['input_audio'].to(device)
            target_audio = batch['target_audio'].to(device)
            effect_type_label = batch['effect_type_label'].to(device)
            
            # 正解のタイプをone-hotベクトルに変換して条件付けに用いる
            condition_one_hot = F.one_hot(effect_type_label, num_classes=model.hafx.num_effects).float()

            optimizer.zero_grad()
            
            s_hat, _, _ = model(input_audio, afx_type_condition=condition_one_hot)

            # s_hat から不要な「ソース」次元 (dim=1) を削除する
            # [B, 1, C, L] -> [B, C, L]
            s_hat_squeezed = s_hat.squeeze(1)

            # 形状が揃ったテンソルで損失を計算
            loss_mae = criterion_mae(s_hat_squeezed, target_audio)
            
            # STFT損失も同様に修正
            loss_stft = criterion_stft(s_hat_squeezed, target_audio.squeeze(1))
            
            # loss_mae = criterion_mae(s_hat, target_audio)
            # loss_stft = criterion_stft(s_hat.squeeze(1), target_audio.squeeze(1))
            loss = loss_mae + lambda_stft * loss_stft
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = validate_epoch(model, valid_loader, criterion_mae, criterion_stft, None, None, device, stage=1)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

        if best_valid_loss - avg_valid_loss > 1e-5:  # 変化が小さい場合は更新しない
            equalization_count = 0
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), f'./{SAVE_DIR}/best_model_final.pth')
            print(f"  -> Found new best model! Saved to best_model_final.pth")
        else:
            equalization_count += 1
            if equalization_count >= 5:
                print("Early stopping triggered.")
                break

    return train_losses, valid_losses

def train_stage2(model, train_loader, valid_loader, optimizer, criterion_ce, criterion_mse, inv_effect_map, param_dims, device, epochs):
    """第2段階: AFX推定器 (hafx) の学習"""
    model.train()
    # hafxを学習可能に
    for param in model.hafx.parameters():
        param.requires_grad = True
    # hsig (hafx以外の部分) を凍結
    for name, param in model.named_parameters():
        if 'hafx' not in name:
            param.requires_grad = False

    print("\n--- Starting Training Stage 2: Training AFX Estimator (hafx) ---")
    train_losses, valid_losses = [], [] # loss値を記録するリスト
    best_valid_loss = float('inf')
    equalization_count = 0
    
    lambda_param = 1  # パラメータ回帰損失の重み
    print(f"lambda = {lambda_param}")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, valid_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_audio = batch['input_audio'].to(device)
            effect_type_label = batch['effect_type_label'].to(device)
            param_vector = batch['param_vector'].to(device)

            optimizer.zero_grad()
            
            # モデル自身の推定を用いる (afx_type_condition=None)
            _, type_logits, param_predictions = model(input_audio)
            
            # タイプ分類の損失
            loss_ce = criterion_ce(type_logits, effect_type_label)
            
            # パラメータ回帰の損失
            loss_mse = 0
            for type_idx, type_name in inv_effect_map.items():
                mask = (effect_type_label == type_idx)
                if mask.any():
                    p_dim = param_dims[type_name]
                    pred = param_predictions[type_name][mask, :p_dim]
                    true = param_vector[mask, :p_dim]
                    loss_mse += criterion_mse(pred, true)
            loss = loss_ce + lambda_param * loss_mse
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = validate_epoch(model, valid_loader, None, None, criterion_ce, criterion_mse, device, stage=2)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        
        # if avg_valid_loss < best_valid_loss:
        if best_valid_loss - avg_valid_loss > 1e-5:  # 変化が小さい場合は更新しない
            equalization_count = 0
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), f'./{SAVE_DIR}/best_model_final.pth')
            print(f"  -> Found new best model! Saved to best_model_final.pth")
        else:
            equalization_count += 1
            if equalization_count >= 5:
                print("Early stopping triggered.")
                break

    return train_losses, valid_losses  # lossリストを返す

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SunAFXiNet Inference Script")
    parser.add_argument('--stage', type=str, required=True, choices=['1', '2', 'all'], help='Training stage to run (1, 2, or all).')
    parser.add_argument('--train-data-dir', type=str, required=True, help='Path to the training set metadata directory.')
    parser.add_argument('--valid-data-dir', type=str, required=True, help='Path to the validation set metadata directory.')
    parser.add_argument('--stage1-model-path', type=str, default='sunafxinet_stage1.pth', help='Path to load/save the best Stage 1 model.')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(DATASET_DIR)
    DRY_SIGNAL_DIR = '../../../dataset/sunafxinet/split_dry_signals/train_dry'

    SAVE_DIR = f'./{datetime.now().strftime("%m%d%H%M")}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # データ生成（必要に応じて実行）
    # print("Generating dataset...")
    # os.makedirs(DATASET_DIR, exist_ok=True)
    # dry_files = glob.glob(os.path.join(DRY_SIGNAL_DIR, '*.wav'))
    # for dry_file in tqdm(dry_files, desc="Generating wet signals"):
    #     create_wet_signal_and_labels(dry_file, DATASET_DIR, EFFECT_TYPES)
    # print("dataset is created")

    print("Using effect:", EFFECT_TYPES)

    hdemucs_config = HDEMUCS_CONFIG
    hdemucs_config['samplerate'] = SAMPLE_RATE

    print("Model initialized.")
    model = SunAFXiNet(hdemucs_config, NUM_EFFECTS, PARAM_DIMS).to(DEVICE)

    # --- データローダーの準備 ---
    print("Loading datasets...")
    train_dataset = AFXChainDataset(metadata_dir=args.train_data_dir, effect_map=EFFECT_MAP, param_dims=PARAM_DIMS)
    valid_dataset = AFXChainDataset(metadata_dir=args.valid_data_dir, effect_map=EFFECT_MAP, param_dims=PARAM_DIMS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples.")

    # print("loading dataset...")
    # dataset = AFXChainDataset(metadata_dir=DATASET_DIR, effect_map=EFFECT_MAP, param_dims=PARAM_DIMS)
    # print(dataset.__len__())
    # data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 1. 学習用データセットの総サンプル数を取得
    total_train_samples = len(train_loader.dataset)

    # 2. 最後のバッチに残るサンプル数を計算
    last_batch_size = total_train_samples % BATCH_SIZE

    # 3. 結果を表示
    # 【修正】MultiResolutionSTFTLossのパラメータを補完しました。
    # 一般的なオーディオ分離タスクで使われる標準的な設定です。
    # 論文のwin_length=8192はモデルのエンコーダ用であり、損失関数は複数の解像度で評価するのが一般的です。
    if args.stage in ['1', 'all']:
        optimizer_stage1 = optim.Adam([p for name, p in model.named_parameters() if 'hafx' not in name], lr=LR_STAGE1)
        criterion_mae_s1 = nn.L1Loss().to(DEVICE)
        criterion_stft_s1 = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096], 
            hop_sizes=[256, 512, 1024], 
            win_lengths=[1024, 2048, 4096],
            sample_rate=SAMPLE_RATE
        ).to(DEVICE)
    
        train_losses_s1, valid_losses_s1 = train_stage1(model, train_loader, valid_loader, optimizer_stage1, criterion_mae_s1, criterion_stft_s1, LAMBDA_STFT, DEVICE, EPOCHS_STAGE1)
        torch.save(model.state_dict(), f'./{SAVE_DIR}/{args.stage1_model_path}_{datetime.now().strftime("%m%d%H%M")}.pth')
        print("Saved Stage 1 model.")
        
        # Stage 1のlossグラフを保存
        plot_losses(train_losses_s1, valid_losses_s1, "Stage 1", save_path=f"./{SAVE_DIR}/stage1_loss_curve.png")

    elif args.stage == "2":
        # Stage 1で学習済みの重みを読み込み
        try:
            model.load_state_dict(torch.load(args.stage1_model_path, map_location=DEVICE))
            print("Loaded Stage 1 model weights.")
        except FileNotFoundError:
            print("Warning: sunafxinet_stage1.pth not found. Starting Stage 2 from random initialization.")
        
        optimizer_stage2 = optim.Adam(model.hafx.parameters(), lr=LR_STAGE2)
        criterion_ce_s2 = nn.CrossEntropyLoss().to(DEVICE)
        criterion_mse_s2 = nn.MSELoss().to(DEVICE)

        train_losses_s2, valid_losses_s2 = train_stage2(model, train_loader, valid_loader, optimizer_stage2, criterion_ce_s2, criterion_mse_s2, DEVICE, EPOCHS_STAGE2)
        torch.save(model.state_dict(), f'./{SAVE_DIR}/sunafxinet_final_{datetime.now().strftime("%m%d%H%M")}.pth')
        print("\nTraining complete. Final model saved as 'sunafxinet_final.pth'.")
        
        # Stage 2のlossグラフを保存
        plot_losses(train_losses_s2, valid_losses_s2, "Stage 2", save_path=f"./{SAVE_DIR}/stage2_loss_curve.png")
