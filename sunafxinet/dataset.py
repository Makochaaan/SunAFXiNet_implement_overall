import os
import glob
import librosa
import soundfile as sf
import numpy as np
import yaml
from tqdm import tqdm
import random


def remove_long_silence(audio, sr, silence_threshold=1.0, top_db=20, hop_length=512):
    """
    1秒以上の無音区間を検出して除去する関数。
    
    Args:
        audio (np.ndarray): 入力音声データ
        sr (int): サンプリングレート
        silence_threshold (float): 無音区間の閾値（秒）
        top_db (int): 無音判定の閾値（dB）
        hop_length (int): フレーム解析のホップ長
        
    Returns:
        np.ndarray: 処理済み音声データ
    """
    # RMSエネルギーを計算
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    
    # 無音閾値を計算（最大RMSから相対的に計算）
    rms_threshold = np.max(rms) * (10 ** (-top_db / 20))
    
    # 無音フレームを検出
    silent_frames = rms < rms_threshold
    
    # フレーム時間を計算
    frame_times = librosa.frames_to_time(np.arange(len(silent_frames)), 
                                        sr=sr, hop_length=hop_length)
    
    # 連続する無音区間を検出
    silent_regions = []
    start_silence = None
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start_silence is None:
            start_silence = i
        elif not is_silent and start_silence is not None:
            # 無音区間の終了
            duration = frame_times[i-1] - frame_times[start_silence]
            if duration >= silence_threshold:
                silent_regions.append((start_silence, i-1))
            start_silence = None
    
    # 最後が無音で終わる場合の処理
    if start_silence is not None:
        duration = frame_times[-1] - frame_times[start_silence]
        if duration >= silence_threshold:
            silent_regions.append((start_silence, len(silent_frames)-1))
    
    if not silent_regions:
        return audio
    
    # 無音区間をサンプル単位に変換
    silence_samples = []
    for start_frame, end_frame in silent_regions:
        start_sample = librosa.frames_to_samples(start_frame, hop_length=hop_length)
        end_sample = librosa.frames_to_samples(end_frame, hop_length=hop_length)
        silence_samples.append((start_sample, end_sample))
    
    # 音声を再構築（長時間無音区間を0.1秒の短い無音に置き換え）
    new_audio_segments = []
    last_end = 0
    replacement_length = int(0.1 * sr)  # 0.1秒の無音
    
    for start_sample, end_sample in silence_samples:
        # 無音区間前の音声を追加
        if start_sample > last_end:
            new_audio_segments.append(audio[last_end:start_sample])
        
        # 短い無音に置き換え
        new_audio_segments.append(np.zeros(replacement_length))
        last_end = end_sample
    
    # 最後の部分を追加
    if last_end < len(audio):
        new_audio_segments.append(audio[last_end:])
    
    processed_audio = np.concatenate(new_audio_segments) if new_audio_segments else audio
    
    # 処理結果をログ出力
    total_removed = sum(end - start for start, end in silence_samples) / sr
    # if total_removed > 0:
        # print(f"  Removed {total_removed:.2f}s of long silence from {len(silent_regions)} regions")
    
    return processed_audio


def find_guitar_stems_from_slakh(slakh_base_dir):
    """
    slakh2100_flac_reduxデータセットからギターのstemファイルを検索する。
    
    Args:
        slakh_base_dir (str): slakh2100_flac_reduxデータセットのベースディレクトリ
        
    Returns:
        list: ギターのstemファイルのパス一覧
    """
    guitar_files = []
    guitar_keywords = ['guitar', 'Guitar']
    
    # track***フォルダを探索
    # others: 1533
    # omitted: 967
    # train: 3358
    # validation: 729
    # test: 392
    # total: 5975
    # train, validation, test ディレクトリを対象にする（omittedは除外）
    track_patterns = [
        os.path.join(slakh_base_dir, "train", "Track*"),
        os.path.join(slakh_base_dir, "validation", "Track*"),
        os.path.join(slakh_base_dir, "test", "Track*"),
        os.path.join(slakh_base_dir, "omitted", "Track*")  # omittedも含める場合
    ]
    track_dirs = []
    for pattern in track_patterns:
        track_dirs.extend(glob.glob(pattern, recursive=True))
    
    for track_dir in track_dirs:
        if not os.path.isdir(track_dir):
            continue
            
        # metadata.yamlファイルの存在確認
        metadata_path = os.path.join(track_dir, "metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
            
        try:
            # metadata.yamlを読み込み
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
                print(metadata)
            # stemsフォルダのパス
            stems_dir = os.path.join(track_dir, "stems")
            if not os.path.exists(stems_dir):
                continue
            
            # メタデータからギターのstemを特定
            if 'stems' in metadata:
                for stem_id, stem_info in metadata['stems'].items():
                    # inst_classでギターかどうかを判定
                    inst_class = stem_info.get('inst_class', '')
                    midi_program_name = stem_info.get('midi_program_name', '')
                    plugin_name = stem_info.get('plugin_name', '')
                    audio_rendered = stem_info.get('audio_rendered', False)
                    
                    # audio_renderedがtrueでない場合はスキップ
                    if not audio_rendered:
                        continue
                    
                    # ギター楽器の判定
                    is_guitar = (
                        'guitar' in inst_class.lower() or
                        any(keyword.lower() in midi_program_name.lower() 
                            for keyword in guitar_keywords) or
                        any(keyword.lower() in plugin_name.lower() 
                            for keyword in guitar_keywords)
                    )
                    
                    if is_guitar:
                        # stemファイルのパスを構築（stem_idをそのまま使用）
                        stem_file_pattern = os.path.join(stems_dir, f"{stem_id}.flac")
                        if os.path.exists(stem_file_pattern):
                            guitar_files.append(stem_file_pattern)
                            print(f"Found guitar stem: {stem_file_pattern} (class: {inst_class}, program: {midi_program_name})")
                        else:
                            # パターンマッチングでファイルを探す
                            alt_pattern = os.path.join(stems_dir, f"*{stem_id}*.flac")
                            stem_files = glob.glob(alt_pattern)
                            if stem_files:
                                guitar_files.extend(stem_files)
                                print(f"Found guitar stem: {stem_files[0]} (class: {inst_class}, program: {midi_program_name})")
                                
        except Exception as e:
            print(f"Error processing metadata in {track_dir}: {e}")
            continue
    
    return guitar_files


def preprocess_guitarset(input_dir, output_dir, slakh_dir=None, target_sr=48000, segment_duration=10, top_db=20):
    """
    GuitarSetデータセットとSlakh2100データセットを前処理し、学習用のセグメントを生成する。
    全ての階層を再帰的に走査して音声ファイルを処理します。

    Args:
        input_dir (str): 音声ファイル（.wav）が格納されているディレクトリ。
        output_dir (str): 前処理済みのセグメントを保存するディレクトリ。
        slakh_dir (str, optional): slakh2100_flac_reduxデータセットのディレクトリ。
        target_sr (int): 目標サンプリングレート。
        segment_duration (int): 各セグメントの長さ（秒）。
        top_db (int): 無音区間をトリミングする際の閾値。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_audio_files = []
    
    # 1. 既存のWAVファイルを検索
    print("Searching for WAV files...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                all_audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_audio_files)} WAV files.")
    
    # 2. Slakh2100データセットからギターのstemを検索
    if slakh_dir and os.path.exists(slakh_dir):
        print("Searching for guitar stems in Slakh2100 dataset...")
        guitar_stems = find_guitar_stems_from_slakh(slakh_dir)
        all_audio_files.extend(guitar_stems)
        print(f"Found {len(guitar_stems)} guitar stems from Slakh2100.")
    
    print(f"Total audio files to process: {len(all_audio_files)}")
    
    if len(all_audio_files) == 0:
        print("No audio files found in the specified directories.")
        return

    segment_length = target_sr * segment_duration
    total_segments = 0

    for audio_file in tqdm(all_audio_files, desc="Preprocessing Audio Files"):
        try:
            # 1. 音声の読み込みとリサンプリング
            # FLACファイルもlibrosaで読み込める
            audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)

            # 2. 無音区間のトリミング
            audio_trimmed = remove_long_silence(audio, sr, silence_threshold=1.0, top_db=top_db)

            # 3. 音声の長さをチェック
            if len(audio_trimmed) < segment_length:
                # print(f"Skipping {audio_file}: too short ({len(audio_trimmed)/target_sr:.2f}s < {segment_duration}s)")
                continue

            # 4. セグメントを抽出
            num_segments = len(audio_trimmed) // segment_length
            
            i = random.randint(0, num_segments - 1)  # ランダムにセグメントを選択
            start_sample = i * segment_length
            end_sample = start_sample + segment_length
            segment = audio_trimmed[start_sample:end_sample]

            # ファイル名の生成と保存
            # データセット種別を識別
            if slakh_dir and slakh_dir in audio_file:
                # Slakh2100の場合
                rel_path = os.path.relpath(audio_file, slakh_dir)
                base_name = f"slakh_{os.path.splitext(rel_path.replace(os.sep, '_'))[0]}"
            else:
                # 既存のWAVファイルの場合
                rel_path = os.path.relpath(audio_file, input_dir)
                base_name = os.path.splitext(rel_path.replace(os.sep, '_'))[0]
            
            # _micサフィックスを除去（GuitarSetの場合）
            if base_name.endswith('_mic'):
                base_name = base_name[:-4]
            
            output_filename = f"{base_name}_segment_{i}.wav"
            output_path = os.path.join(output_dir, output_filename)
            sf.write(output_path, segment, target_sr)
            total_segments += 1

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print(f"Successfully created {total_segments} segments from {len(all_audio_files)} audio files.")


# --- 使用例 ---
guitarset_audio_dir = '/home/depontes25/Desktop/Research/dataset/sunafxinet'
slakh_data_dir = '/home/depontes25/Desktop/Research/dataset/sunafxinet/slakh2100_flac_redux'  # Slakh2100データセットのパス
processed_dry_dir = '../../../dataset/sunafxinet/dry_signal/'

preprocess_guitarset(guitarset_audio_dir, processed_dry_dir, slakh_dir=slakh_data_dir)
