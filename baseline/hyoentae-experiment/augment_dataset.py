"""
오프라인 데이터 증강 스크립트

Train 데이터셋을 물리적으로 증강하여 새로운 이미지 파일 생성

사용법:
    # 기본 증강 (각 이미지당 5개 증강)
    python augment_dataset.py augmentation=strong augment.n_augmentations=5

    # 커스텀 설정
    python augment_dataset.py augmentation=strong augment.n_augmentations=10 augment.output_suffix=_aug10x

결과:
    - data/train_augmented/: 증강된 이미지 저장
    - data/train_augmented.csv: 증강된 데이터셋 CSV (원본 + 증강)
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

from src.data.transforms import create_transforms_from_config


def create_augmentation_pipeline(cfg: DictConfig) -> A.Compose:
    """
    데이터 증강 파이프라인 생성 (ToTensorV2 제외)

    Args:
        cfg: Hydra config

    Returns:
        Albumentations Compose (이미지 저장용, Tensor 변환 제외)
    """
    # 기본 transform 생성 (ToTensorV2 포함)
    full_transform = create_transforms_from_config(cfg, mode='train')

    # ToTensorV2와 Normalize 제거 (이미지 저장을 위해)
    # Albumentations Compose의 transforms 리스트에서 마지막 2개 제거
    aug_transforms = full_transform.transforms[:-2]  # Normalize, ToTensorV2 제거

    return A.Compose(aug_transforms)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    메인 데이터 증강 함수

    Args:
        cfg: Hydra config
    """
    print("\n" + "="*70)
    print("🎨 오프라인 데이터 증강 시작")
    print("="*70)
    print(OmegaConf.to_yaml(cfg.get('augment', {})))
    print("="*70 + "\n")

    # 설정 읽기
    data_dir = Path(cfg.data.data_dir)
    train_csv = data_dir / cfg.data.train_csv
    train_img_dir = data_dir / cfg.data.train_dir

    # 증강 설정
    augment_cfg = cfg.get('augment', {})
    n_augmentations = augment_cfg.get('n_augmentations', 5)  # 각 이미지당 증강 개수
    output_suffix = augment_cfg.get('output_suffix', '_augmented')
    include_original = augment_cfg.get('include_original', True)  # 원본 포함 여부

    # 출력 디렉토리
    output_img_dir = data_dir / f"train{output_suffix}"
    output_csv = data_dir / f"train{output_suffix}.csv"

    output_img_dir.mkdir(exist_ok=True, parents=True)

    print(f"📂 입력:")
    print(f"   CSV: {train_csv}")
    print(f"   Images: {train_img_dir}")
    print(f"\n📂 출력:")
    print(f"   CSV: {output_csv}")
    print(f"   Images: {output_img_dir}")
    print(f"\n⚙️  설정:")
    print(f"   증강 개수: {n_augmentations}개/이미지")
    print(f"   원본 포함: {include_original}")
    print(f"   Augmentation: {cfg.get('augmentation', {}).get('name', 'unknown')}\n")

    # CSV 읽기
    df = pd.read_csv(train_csv)
    print(f"✅ 원본 데이터: {len(df)}개\n")

    # Augmentation 파이프라인 생성
    try:
        aug_pipeline = create_augmentation_pipeline(cfg)
        print(f"✅ Augmentation 파이프라인 생성 완료")
        print(f"   적용되는 변환: {len(aug_pipeline.transforms)}개\n")
    except Exception as e:
        print(f"❌ Augmentation 파이프라인 생성 실패: {e}")
        return

    # 증강된 데이터 저장용 리스트
    augmented_data = []

    # 원본 데이터 포함
    if include_original:
        print("📋 원본 이미지 복사 중...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="원본 복사"):
            img_id = row['ID']
            target = row['target']

            # 원본 이미지 읽기
            img_path = train_img_dir / img_id
            if not img_path.exists():
                print(f"⚠️  이미지 없음: {img_path}")
                continue

            # 원본 이미지 복사
            image = cv2.imread(str(img_path))
            output_path = output_img_dir / img_id
            cv2.imwrite(str(output_path), image)

            # CSV에 추가
            augmented_data.append({
                'ID': img_id,
                'target': target
            })

    # 데이터 증강
    print(f"\n🎨 데이터 증강 중 ({n_augmentations}개/이미지)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="증강 진행"):
        img_id = row['ID']
        target = row['target']

        # 이미지 읽기
        img_path = train_img_dir / img_id
        if not img_path.exists():
            print(f"⚠️  이미지 없음: {img_path}")
            continue

        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # N개의 증강 버전 생성
        for aug_idx in range(n_augmentations):
            try:
                # Augmentation 적용
                augmented = aug_pipeline(image=image_rgb)
                aug_image = augmented['image']

                # RGB → BGR (OpenCV 저장용)
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

                # 파일명 생성 (확장자 분리)
                img_stem = Path(img_id).stem
                img_ext = Path(img_id).suffix
                aug_img_id = f"{img_stem}_aug{aug_idx}{img_ext}"

                # 이미지 저장
                output_path = output_img_dir / aug_img_id
                cv2.imwrite(str(output_path), aug_image_bgr)

                # CSV에 추가
                augmented_data.append({
                    'ID': aug_img_id,
                    'target': target
                })

            except Exception as e:
                print(f"⚠️  증강 실패 ({img_id}, aug{aug_idx}): {e}")
                continue

    # 증강된 CSV 저장
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print("✅ 데이터 증강 완료!")
    print(f"{'='*70}")
    print(f"📊 결과:")
    print(f"   원본 데이터: {len(df)}개")
    print(f"   증강 후 데이터: {len(augmented_df)}개")
    print(f"   증가율: {len(augmented_df) / len(df):.1f}배")
    print(f"\n📂 저장 위치:")
    print(f"   CSV: {output_csv}")
    print(f"   Images: {output_img_dir}/")
    print(f"\n💡 사용 방법:")
    print(f"   python train.py data.train_csv=train{output_suffix}.csv data.train_dir=train{output_suffix}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
