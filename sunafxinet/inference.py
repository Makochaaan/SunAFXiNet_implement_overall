import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from einops import rearrange
import os
import json
import argparse

from sunafxinet import SunAFXiNet
from datetime import datetime

# ============================================================================
# 推論関数 (Inference Function)
# ============================================================================

PARAM_NAMES = {
    'Distortion': ['drive_db'],
    'Chorus': ['rate_hz', 'depth', 'mix'],
    'Delay': ['delay_seconds', 'feedback', 'mix'],
    'Reverb': ['room_size', 'damping', 'wet_level', 'dry_level']
}

PARAM_RANGES = {
    'Distortion': {'drive_db': (10, 40)},
    'Chorus': {'rate_hz': (0.5, 5.0), 'depth': (0.2, 0.8), 'mix': (0.1, 0.5)},
    'Delay': {'delay_seconds': (0.1, 0.8), 'feedback': (0.1, 0.6), 'mix': (0.1, 0.5)},
    'Reverb': {'room_size': (0.1, 0.9), 'damping': (0.1, 0.9), 'wet_level': (0.1, 0.5), 'dry_level': (0.5, 0.9)}
}


def de_normalize_param(value, min_val, max_val):
    """[0, 1]の値を元の範囲に逆正規化する"""
    return value * (max_val - min_val) + min_val

def iterative_inference(model, wet_signal, effect_map, device, max_iterations=2):
    """学習済みモデルを用いて、ウェット信号からドライ信号とエフェクトチェーンを推定します。"""
    model.eval()
    current_signal = torch.from_numpy(wet_signal).unsqueeze(0).unsqueeze(0).to(device)
    estimated_chain_reversed = []
    inv_effect_map = {v: k for k, v in effect_map.items()}

    param_ranges = {
        'Distortion': {'drive_db': (10, 40)},
        'Chorus': {'rate_hz': (0.5, 5.0), 'depth': (0.2, 0.8), 'mix': (0.1, 0.5)},
        'Delay': {'delay_seconds': (0.1, 0.8), 'feedback': (0.1, 0.6), 'mix': (0.1, 0.5)},
        'Reverb': {'room_size': (0.1, 0.9), 'damping': (0.1, 0.9), 'wet_level': (0.1, 0.5), 'dry_level': (0.5, 0.9)}
    }
    
    with torch.no_grad():
        for i in range(max_iterations):
            print(f"Running inference iteration {i+1}/{max_iterations}...")
            s_hat_tensor, type_logits, param_predictions = model(current_signal)
            pred_type_idx = torch.argmax(type_logits, dim=1).item()
            
            if pred_type_idx >= len(inv_effect_map):
                print("Stopping: 'no effect' class was predicted or index is out of bounds.")
                break
                
            prev_energy = torch.mean(current_signal**2)
            new_energy = torch.mean(s_hat_tensor**2)
            print(f" > Signal energy (prev): {prev_energy.item():.6f}, (new): {new_energy.item():.6f}")
            print(torch.abs(prev_energy - new_energy) / prev_energy)
            if torch.abs(prev_energy - new_energy) / prev_energy < 1e-6:
                print("Stopping: Signal energy change is negligible.")
                break
                
            pred_type_name = inv_effect_map[pred_type_idx]
            # モデルは[0,1]の正規化された値を予測する
            pred_params_normalized = np.atleast_1d(param_predictions[pred_type_name].squeeze().cpu().numpy())
            
            # パラメータ名と値を対応付けながら、逆正規化する
            param_names = PARAM_NAMES.get(pred_type_name, [])
            de_normalized_params_dict = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = PARAM_RANGES[pred_type_name][param_name]
                # 予測された正規化値を元のスケールに戻す
                real_value = de_normalize_param(pred_params_normalized[j], min_val, max_val)
                de_normalized_params_dict[param_name] = real_value

            # pedalboardが直接使える辞書を保存する
            estimated_chain_reversed.append({
                'type': pred_type_name, 
                'params': de_normalized_params_dict
            })
            print(f" > Detected: {pred_type_name}")
            
            current_signal = s_hat_tensor.squeeze(1)
            
    final_dry_signal = current_signal.squeeze(0).squeeze(0).cpu().numpy()
    return list(reversed(estimated_chain_reversed)), final_dry_signal

# ============================================================================
# メイン実行ブロック (Main Execution Block)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SunAFXiNet Inference Script")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input wet audio (.wav) file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output dry audio (.wav) file.')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate of the audio.')
    args = parser.parse_args()

    # --- デバイス設定 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 設定 (学習時と完全に一致させる) ---
    ### 重要: 以下の設定は、モデルの学習時に使用したものと完全に一致させる必要があります ###
    # EFFECT_TYPES = ['Distortion', 'Chorus', 'Delay', 'Reverb']
    EFFECT_TYPES = ['Distortion','Reverb']
    EFFECT_MAP = {name: i for i, name in enumerate(EFFECT_TYPES)}
    PARAM_DIMS = {'Distortion': 1, 'Chorus': 3, 'Delay': 3, 'Reverb': 4}
    NUM_EFFECTS = len(EFFECT_TYPES)
    
    hdemucs_config = {
        'audio_channels': 1, 'channels': 48, 'growth': 2, 'nfft': 4096,
        'cac': True, 'depth': 5, 'rewrite': True, 'dconv_mode': 3,
        't_layers': 4, 'samplerate': args.sr, 'segment': 10.0,
        't_heads': 8
    }
    
    # --- モデルのロード ---
    print(f"Loading model from {args.model_path}...")
    model = SunAFXiNet(hdemucs_config, NUM_EFFECTS, PARAM_DIMS)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("Model loaded successfully.")

    # --- 音声ファイルのロード ---
    print(f"Loading audio from {args.input}...")
    try:
        wet_signal, sr = librosa.load(args.input, sr=args.sr, mono=True)
        print(f"Audio loaded successfully. Duration: {len(wet_signal)/sr:.2f} seconds.")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # --- 推論の実行 ---
    print("\nStarting inference process...")
    estimated_chain, dry_signal = iterative_inference(model, wet_signal, EFFECT_MAP, DEVICE)
    print("Inference complete.")

    # --- 結果の保存 ---
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 音声ファイルの保存
    try:
        output_base, output_ext = os.path.splitext(args.output)
        output_filename = f"{output_base}_{datetime.now().strftime('%m%d%H%M')}{output_ext}"
        sf.write(output_filename, dry_signal, args.sr)
        print(f"\nSuccessfully saved processed audio to: {args.output}")
    except Exception as e:
        print(f"Error saving audio file: {e}")

    # エフェクトチェーンの保存
    chain_output_path = os.path.splitext(args.output)[0] + '_effects_' + datetime.now().strftime("%m%d%H%M") + '.json'
    try:
        with open(chain_output_path, 'w') as f:
            json.dump(estimated_chain, f, indent=4)
        print(f"Successfully saved estimated effect chain to: {chain_output_path}")
        
        print("\n--- Estimated Effect Chain (in order of application) ---")
        if estimated_chain:
            for i, effect in enumerate(estimated_chain):
                print(f"{i+1}. Type: {effect['type']}, Params: {effect['params']}")
        else:
            print("No effects were detected.")
        print("---------------------------------------------------------")

    except Exception as e:
        print(f"Error saving effect chain JSON: {e}")


if __name__ == '__main__':
    main()
