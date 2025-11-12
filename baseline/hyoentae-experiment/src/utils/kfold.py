"""
K-Fold Cross Validation 모듈

전체 학습 데이터를 K개의 fold로 나누어 K번 학습합니다.
각 fold에서 한 번씩 validation set으로 사용되며, 나머지는 train set으로 사용됩니다.

최종적으로 K개의 모델이 생성되며, 앙상블이나 평균 점수 계산에 사용할 수 있습니다.
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path


def create_kfold_splits(
    train_csv: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    K-Fold 분할 생성 (Stratified)

    각 fold에서 클래스 비율을 유지하면서 train/val을 분할합니다.

    Args:
        train_csv: 학습 CSV 파일 경로 (ID, target 컬럼 필요)
        n_splits: Fold 개수 (기본 5)
        shuffle: 셔플 여부
        random_state: 랜덤 시드

    Returns:
        [(train_indices, val_indices), ...] 리스트
        각 fold별로 train/val 인덱스를 반환

    예시:
        >>> splits = create_kfold_splits("train.csv", n_splits=5)
        >>> for fold_idx, (train_idx, val_idx) in enumerate(splits):
        >>>     print(f"Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
    """
    # CSV 읽기
    df = pd.read_csv(train_csv)

    # StratifiedKFold 생성 (클래스 비율 유지)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    # 분할 생성
    splits = []
    for train_idx, val_idx in skf.split(df, df['target']):
        splits.append((train_idx, val_idx))

    return splits


def print_kfold_info(splits: List[Tuple[np.ndarray, np.ndarray]], train_csv: str):
    """
    K-Fold 정보 출력

    각 fold별 데이터 개수와 클래스 분포를 출력합니다.

    Args:
        splits: create_kfold_splits()로 생성한 분할 리스트
        train_csv: 학습 CSV 파일 경로
    """
    df = pd.read_csv(train_csv)

    print(f"\n{'='*60}")
    print(f"📊 K-Fold Cross Validation 정보")
    print(f"{'='*60}")
    print(f"전체 데이터: {len(df)}개")
    print(f"Fold 개수: {len(splits)}")
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_targets = df.iloc[train_idx]['target']
        val_targets = df.iloc[val_idx]['target']

        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_idx)}개")
        print(f"  Val: {len(val_idx)}개")
        print(f"  Train 클래스 분포: {dict(train_targets.value_counts().sort_index())}")
        print(f"  Val 클래스 분포: {dict(val_targets.value_counts().sort_index())}")
        print()

    print(f"{'='*60}\n")


def get_fold_save_dir(base_dir: str, fold_idx: int) -> Path:
    """
    Fold별 저장 디렉토리 경로 생성

    Args:
        base_dir: 기본 출력 디렉토리 (예: "outputs/2025-11-03/12-00-00")
        fold_idx: Fold 인덱스

    Returns:
        Fold별 저장 디렉토리 Path 객체

    예시:
        >>> save_dir = get_fold_save_dir("outputs/2025-11-03/12-00-00", 0)
        >>> print(save_dir)  # outputs/2025-11-03/12-00-00/fold_0
    """
    return Path(base_dir) / f"fold_{fold_idx}"


def save_kfold_summary(
    save_dir: str,
    fold_results: List[dict]
):
    """
    K-Fold 전체 결과 요약 저장

    각 fold별 결과와 평균을 계산하여 저장합니다.

    Args:
        save_dir: 저장 디렉토리
        fold_results: 각 fold별 결과 리스트
            [{"fold": 0, "best_epoch": 10, "val_f1": 0.85, ...}, ...]
    """
    import json

    save_path = Path(save_dir) / "kfold_summary.json"

    # 평균 계산
    avg_val_f1 = np.mean([r['val_f1'] for r in fold_results])
    avg_train_f1 = np.mean([r.get('train_f1', 0) for r in fold_results])
    avg_best_epoch = np.mean([r['best_epoch'] for r in fold_results])

    summary = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "average": {
            "val_macro_f1": float(avg_val_f1),
            "train_macro_f1": float(avg_train_f1),
            "best_epoch": float(avg_best_epoch),
        }
    }

    # JSON 저장
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ K-Fold 전체 결과 요약")
    print(f"{'='*60}")
    print(f"평균 Val Macro F1: {avg_val_f1:.4f}")
    print(f"평균 Train Macro F1: {avg_train_f1:.4f}")
    print(f"평균 Best Epoch: {avg_best_epoch:.1f}")
    print(f"\n각 Fold별 결과:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Val F1={result['val_f1']:.4f}, "
              f"Train F1={result.get('train_f1', 0):.4f}, "
              f"Best Epoch={result['best_epoch']}")
    print(f"\n요약 저장: {save_path}")
    print(f"{'='*60}\n")
