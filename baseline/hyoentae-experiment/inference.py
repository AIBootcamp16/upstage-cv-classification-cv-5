"""
Inference Script

TTA (Test Time Augmentation)를 사용한 추론 및 submission.csv 생성

사용법:
    # 기본 추론 (TTA 사용)
    python inference.py checkpoint=outputs/2025-11-02/12-00-00/best.pth

    # TTA 없이 추론
    python inference.py checkpoint=outputs/2025-11-02/12-00-00/best.pth use_tta=false

    # Config 오버라이드
    python inference.py checkpoint=best.pth model=efficientnet_b0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

from src.models.classifier import create_model_from_config
from src.data.dataset import create_test_dataset, create_test_dataloader
from src.data.transforms import create_transforms_from_config, get_tta_transforms
from src.utils.checkpoint import load_model_for_inference, find_latest_checkpoint


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    tta_transforms: list,
    device: str
) -> np.ndarray:
    """
    TTA (Test Time Augmentation)로 예측

    여러 augmentation을 적용한 이미지들의 예측을 평균냅니다.

    Args:
        model: PyTorch 모델
        image: 원본 이미지 (NumPy 배열, RGB)
        tta_transforms: TTA 변환 리스트
        device: 디바이스

    Returns:
        평균 예측 확률 (shape: [num_classes])
    """
    predictions = []

    for transform in tta_transforms:
        # Augmentation 적용
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)  # [1, C, H, W]

        # 예측
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]  # [num_classes]
        predictions.append(prob)

    # 평균 확률
    avg_prob = np.mean(predictions, axis=0)

    return avg_prob


@torch.no_grad()
def inference_with_dataloader(
    model: torch.nn.Module,
    test_loader,
    device: str,
    use_tta: bool = False,
    tta_transforms: list = None,
    test_img_dir: str = None,
) -> tuple:
    """
    DataLoader를 사용한 배치 추론

    Args:
        model: PyTorch 모델
        test_loader: Test DataLoader
        device: 디바이스
        use_tta: TTA 사용 여부
        tta_transforms: TTA 변환 리스트 (TTA 사용 시 필요)
        test_img_dir: Test 이미지 디렉토리 (TTA 사용 시 필요)

    Returns:
        (img_ids, predictions) 튜플
    """
    model.eval()

    all_img_ids = []
    all_predictions = []

    if use_tta:
        print("🔄 TTA (Test Time Augmentation) 사용")
        print(f"   TTA 버전 수: {len(tta_transforms)}")

        # TTA는 DataLoader 사용 안 하고 개별 이미지 처리
        test_img_dir = Path(test_img_dir)

        # sample_submission.csv에서 이미지 ID 읽기
        for images, img_ids in tqdm(test_loader, desc="Inference (TTA)"):
            for i, img_id in enumerate(img_ids):
                # 이미지 파일 경로 결정
                # ID에 이미 확장자가 포함되어 있는지 확인
                if img_id.endswith('.jpg') or img_id.endswith('.png'):
                    # ID에 이미 확장자 포함됨
                    img_path = test_img_dir / img_id
                else:
                    # 확장자 없음, jpg 또는 png 시도
                    img_path_jpg = test_img_dir / f"{img_id}.jpg"
                    img_path_png = test_img_dir / f"{img_id}.png"

                    if img_path_jpg.exists():
                        img_path = img_path_jpg
                    elif img_path_png.exists():
                        img_path = img_path_png
                    else:
                        print(f"⚠️  이미지를 찾을 수 없습니다: {img_id}")
                        continue

                # 파일 존재 확인
                if not img_path.exists():
                    print(f"⚠️  이미지를 찾을 수 없습니다: {img_path}")
                    continue

                # 이미지 읽기
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # TTA로 예측
                avg_prob = predict_with_tta(model, image, tta_transforms, device)
                prediction = avg_prob.argmax()

                all_img_ids.append(img_id)
                all_predictions.append(prediction)

    else:
        print("📊 기본 추론 (TTA 없음)")

        for images, img_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)

            # 예측
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_img_ids.extend(img_ids)
            all_predictions.extend(predictions)

    return all_img_ids, all_predictions


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    메인 추론 함수

    Args:
        cfg: Hydra config
    """
    # Hydra struct 모드 해제 (새로운 키 추가 가능하도록)
    OmegaConf.set_struct(cfg, False)

    # Config 출력
    print("\n" + "="*60)
    print("⚙️  Inference Config")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")

    # 체크포인트 경로
    checkpoint_path = cfg.get('checkpoint', None)

    if checkpoint_path is None:
        # 자동으로 최신 체크포인트 찾기
        print("📦 체크포인트 경로가 지정되지 않음 → 최신 실험 자동 검색")
        try:
            checkpoint_path = find_latest_checkpoint(output_dir="outputs")
        except FileNotFoundError as e:
            print(str(e))
            raise
    else:
        print(f"📦 체크포인트: {checkpoint_path}")

    # 디바이스
    if cfg.get('device', 'cuda') == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("✅ CPU 사용")

    # 데이터 경로
    data_dir = Path(cfg.data.data_dir)
    test_csv = data_dir / cfg.data.get('test_csv', 'sample_submission.csv')
    test_img_dir = data_dir / cfg.data.get('test_dir', 'test')

    print(f"\n📂 데이터 경로:")
    print(f"   CSV: {test_csv}")
    print(f"   Images: {test_img_dir}\n")

    # 체크포인트에서 모델 config 읽기 (있으면)
    print("📦 체크포인트에서 모델 정보 확인 중...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_config' in checkpoint:
        # 체크포인트에 모델 config가 저장되어 있음 (자동 감지)
        saved_model_config = checkpoint['model_config']
        model_name = saved_model_config.get('name', 'unknown')
        print(f"✅ 체크포인트에서 모델 정보 발견: {model_name}")

        # 저장된 config로 모델 생성
        temp_cfg = OmegaConf.create({'model': saved_model_config, 'data': cfg.data})
        model = create_model_from_config(temp_cfg, num_classes=cfg.data.num_classes)
    else:
        # 모델 config가 없음 (구 버전 체크포인트)
        print("⚠️  체크포인트에 모델 정보 없음 → config.yaml에서 모델 로드")
        print(f"   사용 모델: {cfg.model.name}")
        model = create_model_from_config(cfg, num_classes=cfg.data.num_classes)

    # 체크포인트 로드
    model = load_model_for_inference(model, checkpoint_path, device)

    # TTA 사용 여부
    use_tta = cfg.get('use_tta', True)

    if use_tta:
        # TTA 변환 생성
        tta_transforms = get_tta_transforms(img_size=cfg.model.get('img_size', 224))

        # TTA는 DataLoader 사용 안 함 (개별 이미지 처리)
        # 하지만 img_id를 읽기 위해 DataLoader 사용
        test_transform = create_transforms_from_config(cfg, mode='test')
        test_dataset = create_test_dataset(
            csv_path=str(test_csv),
            img_dir=str(test_img_dir),
            transform=test_transform,
        )
        test_loader = create_test_dataloader(
            test_dataset,
            batch_size=1,  # TTA는 1개씩 처리
            num_workers=0,
        )

        # 추론
        img_ids, predictions = inference_with_dataloader(
            model=model,
            test_loader=test_loader,
            device=device,
            use_tta=True,
            tta_transforms=tta_transforms,
            test_img_dir=str(test_img_dir),
        )

    else:
        # 기본 추론 (TTA 없음)
        test_transform = create_transforms_from_config(cfg, mode='test')
        test_dataset = create_test_dataset(
            csv_path=str(test_csv),
            img_dir=str(test_img_dir),
            transform=test_transform,
        )
        test_loader = create_test_dataloader(
            test_dataset,
            batch_size=cfg.train.get('batch_size', 32),
            num_workers=cfg.data.get('num_workers', 4),
        )

        # 추론
        img_ids, predictions = inference_with_dataloader(
            model=model,
            test_loader=test_loader,
            device=device,
            use_tta=False,
        )

    # Submission DataFrame 생성
    submission = pd.DataFrame({
        'ID': img_ids,
        'target': predictions
    })

    # sample_submission.csv의 순서 맞추기
    original_submission = pd.read_csv(test_csv)
    submission = original_submission[['ID']].merge(submission, on='ID', how='left')

    # 누락된 예측값 확인
    if submission['target'].isna().any():
        print(f"⚠️  경고: {submission['target'].isna().sum()}개 이미지 예측 실패")

    # Submission 저장
    submission_name = cfg.get('submission_name', 'submission.csv')
    submission_path = submission_name
    submission.to_csv(submission_path, index=False)

    print(f"\n✅ 추론 완료!")
    print(f"   전체 샘플: {len(submission)}")
    print(f"   Submission 저장: {submission_path}")
    print(f"\n📊 예측 분포:")
    print(submission['target'].value_counts().sort_index())


if __name__ == "__main__":
    main()
