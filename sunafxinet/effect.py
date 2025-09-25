import pedalboard
import soundfile as sf
import numpy as np
import os
import json
import random
from tqdm import tqdm
import glob
import itertools
import shutil

from constant import EFFECT_TYPES

# エフェクトの定義

def get_random_effect_params(effect_type):
    """ 指定されたエフェクトタイプのランダムなパラメータを返す """
    if effect_type == 'Distortion':
        return {'drive_db': random.uniform(10.0, 40.0)}
    elif effect_type == 'Chorus':
        return {
            'rate_hz': random.uniform(0.5, 5.0),
            'depth': random.uniform(0.2, 0.8),
            'mix': random.uniform(0.1, 0.5)
        }
    elif effect_type == 'Delay':
        return {
            'delay_seconds': random.uniform(0.1, 0.8),
            'feedback': random.uniform(0.1, 0.6),
            'mix': random.uniform(0.1, 0.5)
        }
    elif effect_type == 'Reverb':
        return {
            'room_size': random.uniform(0.1, 0.9),
            'damping': random.uniform(0.1, 0.9),
            'wet_level': random.uniform(0.1, 0.5),
            'dry_level': random.uniform(0.5, 0.9)
        }
    return {}

def generate_afx_chains(effect_types, max_chain_length=4):
    """
    64パターンのエフェクトチェーン（順列）を事前に生成する。
    """
    all_chains = []
    for i in range(1, max_chain_length + 1):
        # i個のエフェクトを選ぶ順列を生成
        for p in itertools.permutations(effect_types, i):
            chain_def = []
            for effect_type in p:
                params = get_random_effect_params(effect_type)
                chain_def.append({'type': effect_type, 'params': params})
            all_chains.append(chain_def)
    return all_chains

def apply_chain_and_create_labels(dry_audio_path, output_dir, chain_definition, chain_id, sr=48000):
    """
    1つのドライ信号に、事前に定義された1つのエフェクトチェーンを適用する。
    """
    try:
        dry_audio, sr_read = sf.read(dry_audio_path)
        if sr_read != sr:
            # 必要であればリサンプリング
            dry_audio = librosa.resample(y=dry_audio, orig_sr=sr_read, target_sr=sr)

        base_name = os.path.basename(dry_audio_path).replace('.wav', '')
        # ファイル名が重複しないようにチェーンIDを追加
        output_base_name = f"{base_name}_chain_{chain_id:02d}"

        intermediate_signals = {'dry': dry_audio_path}
        current_signal = dry_audio

        for i, effect_def in enumerate(chain_definition):
            effect_type = effect_def['type']
            params = effect_def['params']
            effect_plugin = getattr(pedalboard, effect_type)(**params)

            board = pedalboard.Pedalboard([effect_plugin])
            current_signal = board(current_signal, sr)

            intermediate_filename = f"{output_base_name}_intermediate_{i}.wav"
            intermediate_path = os.path.join(output_dir, intermediate_filename)
            sf.write(intermediate_path, current_signal, sr)
            intermediate_signals[f'wet_{i}'] = intermediate_path

        # メタデータファイルのパス
        meta_filename = f"{output_base_name}_metadata.json"
        meta_path = os.path.join(output_dir, meta_filename)

        metadata = {
            'dry_signal_path': dry_audio_path,
            'final_wet_signal_path': intermediate_signals.get(f'wet_{len(chain_definition)-1}', dry_audio_path),
            'effect_chain': chain_definition,
            'intermediate_signals': intermediate_signals
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        print(f"Error processing {dry_audio_path} with chain {chain_id}: {e}")

# --- 使用例 ---
# --- メインの実行部分 ---
if __name__ == '__main__':
    types = ["test","train","valid"]
    dry_signal_dirs = ['../../../dataset/sunafxinet/split_dry_signals/test_dry','../../../dataset/sunafxinet/split_dry_signals/train_dry','../../../dataset/sunafxinet/split_dry_signals/valid_dry'] # 前処理済みのドライ信号があるディレクトリ
    dataset_output_dir_base = '../../../dataset/sunafxinet/wet_signal_2way/reb-dis' # 生成されるウェット信号とメタデータを保存するディレクトリ

    if not os.path.exists(dataset_output_dir_base):
        os.makedirs(dataset_output_dir_base)
    else:
        shutil.rmtree(dataset_output_dir_base)

    for n, dry_signal_dir in enumerate(dry_signal_dirs):
        # 1. 64パターンのエフェクトチェーンを事前に生成
        print(f"Generating {len(EFFECT_TYPES)} AFX chains...")
        print("Using effect types:", EFFECT_TYPES)
        afx_chains = generate_afx_chains(EFFECT_TYPES)
        print(f"Generated {len(afx_chains)} chains.")

        # 2. 全てのドライ信号を取得
        dry_files = glob.glob(os.path.join(dry_signal_dir, '*.wav'))
        print(f"Found {len(dry_files)} dry signals in {types[n]}.")
        # 注意: 全ての組み合わせを生成すると膨大なディスク容量と時間が必要です。
        # テスト用にファイル数を制限することをお勧めします。
        # dry_files = dry_files[:10] # 例: 最初の10ファイルのみでテスト
        dataset_output_dir = os.path.join(dataset_output_dir_base, types[n])
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        # 3. 各ドライ信号に64パターンのチェーンをすべて適用
        for dry_file in tqdm(dry_files, desc="Processing dry signals"):
            for i, chain_def in enumerate(tqdm(afx_chains, desc=f"Applying chains to {os.path.basename(dry_file)}", leave=False)):
                apply_chain_and_create_labels(dry_file, dataset_output_dir, chain_def, chain_id=i)

    print("\nDataset generation complete.")
