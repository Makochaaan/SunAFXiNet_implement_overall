import os
import glob
import random
import shutil
from tqdm import tqdm

def split_source_signals(source_dir, output_base_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    ソース信号（ドライ信号）をtrain/valid/testに分割し、ファイルをコピーする。
    論文の比率に近いデフォルト値を使用 (80/10/10)。
    
    Args:
        source_dir (str): 分割元のドライ信号が格納されているディレクトリ。
        output_base_dir (str): 分割後のファイルが保存されるベースディレクトリ。
        train_ratio (float): 学習用データの割合。
        valid_ratio (float): 検証用データの割合。
        test_ratio (float): テスト用データの割合。
    """
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-5:
        raise ValueError("Ratios must sum to 1.")

    # 出力ディレクトリを作成
    train_dir = os.path.join(output_base_dir, 'train_dry')
    valid_dir = os.path.join(output_base_dir, 'valid_dry')
    test_dir = os.path.join(output_base_dir, 'test_dry')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ソース信号のリストを取得
    source_files = glob.glob(os.path.join(source_dir, '*.wav'))
    if not source_files:
        print(f"No source files found in {source_dir}")
        return
        
    # リストをシャッフルしてランダム性を確保
    random.shuffle(source_files)
    
    total_files = len(source_files)
    
    # 分割点を計算
    train_end = int(total_files * train_ratio)
    valid_end = train_end + int(total_files * valid_ratio)
    
    # リストをスライスして分割
    train_files = source_files[:train_end]
    valid_files = source_files[train_end:valid_end]
    test_files = source_files[valid_end:]
    
    print(f"Total source signals: {total_files}")
    print(f"Splitting into -> Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

    # ファイルを対応するディレクトリにコピー
    def copy_files(files, dest_dir):
        for f in tqdm(files, desc=f"Copying to {os.path.basename(dest_dir)}"):
            shutil.copy(f, dest_dir)
            
    copy_files(train_files, train_dir)
    copy_files(valid_files, valid_dir)
    copy_files(test_files, test_dir)
    
    print("\nSplitting complete.")

# --- 使用例 ---
if __name__ == '__main__':
    # 前処理済みのドライ信号がすべて入っているディレクトリ
    SOURCE_DRY_SIGNAL_DIR = '../dry_signal/'
    
    # 分割後のドライ信号が保存される新しいベースディレクトリ
    SPLIT_DRY_SIGNAL_BASE_DIR = '../../../dataset/sunafxinet/split_dry_signals/'
    
    # 論文の比率 (およそ 75% / 9.6% / 15.4%)
    # 4491 / 5975 = 0.7516
    # 572 / 5975 = 0.0957
    # 912 / 5975 = 0.1526
    paper_train_ratio = 4491 / 5975
    paper_valid_ratio = 572 / 5975
    paper_test_ratio = 1.0 - paper_train_ratio - paper_valid_ratio

    split_source_signals(
        SOURCE_DRY_SIGNAL_DIR, 
        SPLIT_DRY_SIGNAL_BASE_DIR,
        train_ratio=paper_train_ratio,
        valid_ratio=paper_valid_ratio,
        test_ratio=paper_test_ratio
    )
