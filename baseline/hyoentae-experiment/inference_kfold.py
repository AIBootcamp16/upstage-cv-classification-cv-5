"""
K-Fold 앙상블 추론 스크립트

여러 fold의 모델을 앙상블하여 최종 예측을 생성합니다.
각 fold의 예측 확률을 평균내어 최종 클래스를 결정합니다.

사용법:
    # 기본 사용 (자동으로 최신 K-Fold 실험 찾기, 모든 fold 사용)
    python inference_kfold.py

    # 특정 K-Fold 실험 디렉토리 지정
    python inference_kfold.py kfold_dir=outputs/2025-11-05/03-32-32

    # TTA 사용 (더 느리지만 성능 향상)
    python inference_kfold.py use_tta=true

    # Val F1 기준 상위 3개 fold만 사용
    python inference_kfold.py top_k_folds=3

    # 특정 fold 번호들만 사용 (예: fold 0, 2, 4)
    python inference_kfold.py 'use_folds=[0,2,4]'

    # 조합 예제
    python inference_kfold.py top_k_folds=3 use_tta=true

장점:
    - 단일 모델보다 안정적이고 높은 성능
    - 각 fold가 다른 validation set으로 학습되어 일반화 성능 향상
    - 특정 fold만 선택하여 추론 속도와 성능 조절 가능
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import cv2

from src.models.classifier import DocumentClassifier
from src.data.dataset import create_test_dataset, create_test_dataloader
from src.data.transforms import create_transforms_from_config, get_tta_transforms


def subclass_to_class(subclass_pred):
    """
    Sub-class 예측을 원래 class로 변환 (38-class → 17-class)

    Args:
        subclass_pred: Sub-class 예측 (0~37)

    Returns:
        원래 class (0~16)
    """
    # Sub-class 범위 체크
    if 30 <= subclass_pred <= 39:
        return 3
    elif 70 <= subclass_pred <= 79:
        return 7
    elif 140 <= subclass_pred <= 143:
        return 14
    else:
        # 나머지는 그대로
        return subclass_pred


def find_latest_kfold_dir(output_dir: str = "outputs") -> Path:
    """
    가장 최신 K-Fold 실험 디렉토리 찾기

    Args:
        output_dir: 출력 디렉토리 (기본값: "outputs")

    Returns:
        K-Fold 실험 디렉토리 경로
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"출력 디렉토리가 없습니다: {output_dir}")

    # 모든 날짜 디렉토리 찾기
    date_dirs = sorted([d for d in output_path.iterdir() if d.is_dir()])

    if not date_dirs:
        raise FileNotFoundError(f"실험 디렉토리가 없습니다: {output_dir}")

    # 최신 날짜 디렉토리 선택
    latest_date_dir = date_dirs[-1]

    # 시간 디렉토리들 찾기
    time_dirs = sorted([d for d in latest_date_dir.iterdir() if d.is_dir()])

    # 각 시간 디렉토리에서 K-Fold 실험 찾기 (kfold_summary.json 존재 여부)
    for time_dir in reversed(time_dirs):  # 최신부터 검색
        if (time_dir / "kfold_summary.json").exists():
            print(f"✅ K-Fold 실험 발견: {time_dir}")
            return time_dir

    raise FileNotFoundError(f"K-Fold 실험을 찾을 수 없습니다. kfold_summary.json이 없습니다.")


def load_kfold_models(kfold_dir: Path, device: str, top_k_folds: int = None, use_folds: list = None) -> list:
    """
    K-Fold 디렉토리에서 fold 모델 로드 (선택적으로 특정 fold만)

    Args:
        kfold_dir: K-Fold 실험 디렉토리
        device: 디바이스 ('cuda' 또는 'cpu')
        top_k_folds: validation f1 기준 상위 k개 fold만 선택 (None이면 모두 사용)
        use_folds: 특정 fold 번호 리스트 (None이면 모두 사용)

    Returns:
        [(model, fold_idx, val_f1, num_classes), ...] 리스트
    """
    # K-Fold summary 읽기
    summary_path = kfold_dir / "kfold_summary.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"\n📊 K-Fold Summary:")
    print(f"   Total Folds: {summary['n_folds']}")
    print(f"   Average Val F1: {summary['average']['val_macro_f1']:.4f}")
    print(f"   Average Train F1: {summary['average']['train_macro_f1']:.4f}\n")

    # Fold 결과 가져오기
    fold_results = summary['fold_results']

    # Fold 선택 로직
    if use_folds is not None:
        # 특정 fold 번호들만 사용
        fold_results = [f for f in fold_results if f['fold'] in use_folds]
        print(f"🎯 특정 Fold 선택: {use_folds}")
        print(f"   선택된 Fold 개수: {len(fold_results)}\n")
    elif top_k_folds is not None:
        # Val F1 기준으로 정렬하고 상위 k개만 선택
        fold_results = sorted(fold_results, key=lambda x: x['val_f1'], reverse=True)[:top_k_folds]
        selected_folds = [f['fold'] for f in fold_results]
        print(f"🏆 상위 {top_k_folds}개 Fold 선택 (Val F1 기준)")
        print(f"   선택된 Fold: {selected_folds}")
        print(f"   Val F1 범위: {fold_results[-1]['val_f1']:.4f} ~ {fold_results[0]['val_f1']:.4f}\n")
    else:
        print(f"📦 전체 {len(fold_results)}개 Fold 사용\n")

    # 각 fold 모델 로드
    models = []

    for fold_info in fold_results:
        fold_idx = fold_info['fold']
        val_f1 = fold_info['val_f1']

        fold_dir = kfold_dir / f"fold_{fold_idx}"
        checkpoint_path = fold_dir / "best.pth"

        if not checkpoint_path.exists():
            print(f"⚠️  Fold {fold_idx} 체크포인트 없음, 건너뜀: {checkpoint_path}")
            continue

        print(f"📦 Fold {fold_idx} 로딩... (Val F1: {val_f1:.4f})")

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Config 복원 (model_config 키로 저장되어 있음)
        if 'model_config' in checkpoint:
            model_cfg = checkpoint['model_config']
        elif 'config' in checkpoint:
            model_cfg = checkpoint['config']['model']
        else:
            raise ValueError(f"체크포인트에 model_config 또는 config가 없습니다: {checkpoint_path}")

        # num_classes 확인: 실제 가중치 shape로부터 추론 (config가 잘못되어 있을 수 있음)
        fc_weight_key = None
        for key in checkpoint['model_state_dict'].keys():
            if 'fc.weight' in key or 'classifier.weight' in key or 'head.weight' in key:
                fc_weight_key = key
                break

        if fc_weight_key:
            actual_num_classes = checkpoint['model_state_dict'][fc_weight_key].shape[0]
            config_num_classes = model_cfg.get('num_classes', 17)

            if actual_num_classes != config_num_classes:
                print(f"   ⚠️  Config num_classes={config_num_classes}, 실제 가중치={actual_num_classes} → 실제 가중치 사용")
                num_classes = actual_num_classes
            else:
                num_classes = config_num_classes
        else:
            num_classes = model_cfg.get('num_classes', 17)

        # 모델 생성
        model = DocumentClassifier(
            model_name=model_cfg['architecture'],
            num_classes=num_classes,
            pretrained=False,
            dropout=model_cfg.get('dropout', 0.3)
        )

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        models.append((model, fold_idx, val_f1, num_classes))

        print(f"   ✅ Fold {fold_idx} 로드 완료 (num_classes={num_classes})")

    if not models:
        raise ValueError(f"로드된 모델이 없습니다. fold_*/best.pth 파일을 확인하세요.")

    print(f"\n✅ 총 {len(models)}개 fold 모델 로드 완료\n")

    return models


@torch.no_grad()
def predict_with_ensemble(
    models: list,
    test_loader,
    device: str,
    use_tta: bool = False,
    tta_transforms: list = None,
    test_img_dir: Path = None,
) -> tuple:
    """
    K-Fold 앙상블 예측 (Sub-class 자동 변환 지원)

    Args:
        models: [(model, fold_idx, val_f1, num_classes), ...] 리스트
        test_loader: Test DataLoader
        device: 디바이스
        use_tta: TTA 사용 여부
        tta_transforms: TTA 변환 리스트
        test_img_dir: Test 이미지 디렉토리

    Returns:
        (img_ids, predictions) 튜플
    """
    all_img_ids = []
    all_predictions = []

    # Sub-class 모델 확인 (첫 번째 모델의 num_classes로 판단)
    num_classes = models[0][3]
    is_subclass_model = (num_classes == 38)

    if is_subclass_model:
        print(f"🏷️  Sub-class 모델 감지 (38-class → 17-class 자동 변환)")

    if use_tta:
        print("🔄 K-Fold 앙상블 + TTA 추론 (가장 강력!)")
        print(f"   모델 수: {len(models)}")
        print(f"   TTA 버전 수: {len(tta_transforms)}")
        print(f"   총 예측 횟수: {len(models)} × {len(tta_transforms)} = {len(models) * len(tta_transforms)}")

        # TTA는 개별 이미지 처리
        for images, img_ids in tqdm(test_loader, desc="K-Fold Ensemble + TTA"):
            for i, img_id in enumerate(img_ids):
                # 이미지 파일 경로
                if img_id.endswith('.jpg') or img_id.endswith('.png'):
                    img_path = test_img_dir / img_id
                else:
                    img_path_jpg = test_img_dir / f"{img_id}.jpg"
                    img_path_png = test_img_dir / f"{img_id}.png"
                    img_path = img_path_jpg if img_path_jpg.exists() else img_path_png

                if not img_path.exists():
                    print(f"⚠️  이미지 없음: {img_path}")
                    continue

                # 이미지 읽기
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 모든 fold + TTA 예측 수집
                ensemble_probs = []

                for model, fold_idx, val_f1, _ in models:
                    for transform in tta_transforms:
                        # Augmentation 적용
                        augmented = transform(image=image)
                        img_tensor = augmented['image'].unsqueeze(0).to(device)

                        # 예측
                        output = model(img_tensor)
                        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                        ensemble_probs.append(prob)

                # 평균 확률
                avg_prob = np.mean(ensemble_probs, axis=0)
                prediction = avg_prob.argmax()

                # Sub-class → class 변환
                if is_subclass_model:
                    prediction = subclass_to_class(prediction)

                all_img_ids.append(img_id)
                all_predictions.append(prediction)

    else:
        print("📊 K-Fold 앙상블 추론 (TTA 없음)")
        print(f"   모델 수: {len(models)}")

        for images, img_ids in tqdm(test_loader, desc="K-Fold Ensemble"):
            images = images.to(device)

            # 모든 fold 모델의 예측 수집
            batch_ensemble_probs = []

            for model, fold_idx, val_f1, _ in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # [batch_size, num_classes]
                batch_ensemble_probs.append(probs)

            # 평균 확률 (모든 fold)
            avg_probs = np.mean(batch_ensemble_probs, axis=0)  # [batch_size, num_classes]
            predictions = avg_probs.argmax(axis=1)

            # Sub-class → class 변환
            if is_subclass_model:
                predictions = [subclass_to_class(p) for p in predictions]

            all_img_ids.extend(img_ids)
            all_predictions.extend(predictions)

    return all_img_ids, all_predictions


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    메인 K-Fold 앙상블 추론 함수
    """
    print("\n" + "="*60)
    print("🔀 K-Fold 앙상블 추론")
    print("="*60)

    # Hydra struct 모드 해제 (새로운 키 추가 가능하도록)
    OmegaConf.set_struct(cfg, False)

    # K-Fold 디렉토리
    kfold_dir = cfg.get('kfold_dir', None)

    if kfold_dir is None:
        print("📂 K-Fold 디렉토리가 지정되지 않음 → 최신 실험 자동 검색")
        try:
            kfold_dir = find_latest_kfold_dir(output_dir="outputs")
        except FileNotFoundError as e:
            print(str(e))
            raise
    else:
        kfold_dir = Path(kfold_dir)
        print(f"📂 K-Fold 디렉토리: {kfold_dir}")

    # 디바이스
    if cfg.get('device', 'cuda') == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("✅ CPU 사용")

    # Fold 선택 옵션
    top_k_folds = cfg.get('top_k_folds', None)
    use_folds = cfg.get('use_folds', None)

    # K-Fold 모델들 로드
    models = load_kfold_models(kfold_dir, device, top_k_folds=top_k_folds, use_folds=use_folds)

    # 데이터 경로
    data_dir = Path(cfg.data.data_dir)
    test_csv = data_dir / cfg.data.get('test_csv', 'sample_submission.csv')
    test_img_dir = data_dir / cfg.data.get('test_dir', 'test')

    print(f"📂 데이터 경로:")
    print(f"   Test CSV: {test_csv}")
    print(f"   Test Images: {test_img_dir}\n")

    # TTA 사용 여부
    use_tta = cfg.get('use_tta', False)

    # 모델 config에서 이미지 크기 가져오기 (첫 번째 모델)
    checkpoint = torch.load(kfold_dir / f"fold_{models[0][1]}" / "best.pth", map_location='cpu', weights_only=False)

    # Config 복원 (model_config 키로 저장되어 있음)
    if 'model_config' in checkpoint:
        img_size = checkpoint['model_config']['input_size']
    elif 'config' in checkpoint:
        img_size = checkpoint['config']['model']['input_size']
    else:
        raise ValueError(f"체크포인트에 model_config 또는 config가 없습니다")

    if use_tta:
        # TTA 변환 생성
        tta_transforms = get_tta_transforms(img_size=img_size)
        print(f"✅ TTA 변환 생성 완료 (버전 수: {len(tta_transforms)})")
    else:
        tta_transforms = None

    # Test transform 생성
    test_transform = create_transforms_from_config(cfg, mode='test')

    # Config에서 img_size 설정 (모델과 일치하도록)
    OmegaConf.set_struct(cfg, False)
    if 'model' not in cfg:
        cfg.model = {}
    cfg.model.img_size = img_size
    OmegaConf.set_struct(cfg, True)

    # Test dataset 생성
    test_dataset = create_test_dataset(
        csv_path=str(test_csv),
        img_dir=str(test_img_dir),
        transform=test_transform
    )

    # Test dataloader 생성
    batch_size = 1 if use_tta else cfg.train.get('batch_size', 32)
    test_loader = create_test_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=cfg.data.get('num_workers', 4)
    )

    print(f"📦 Test Dataset: {len(test_dataset)}개 이미지")
    print(f"   Batch Size: {batch_size}\n")

    # K-Fold 앙상블 추론
    print("🚀 추론 시작...\n")

    img_ids, predictions = predict_with_ensemble(
        models=models,
        test_loader=test_loader,
        device=device,
        use_tta=use_tta,
        tta_transforms=tta_transforms,
        test_img_dir=test_img_dir
    )

    # Submission 생성
    submission_df = pd.DataFrame({
        'ID': img_ids,
        'target': predictions
    })

    # 저장 경로
    submission_name = cfg.get('submission_name', 'submission_kfold.csv')
    if use_tta:
        submission_name = submission_name.replace('.csv', '_tta.csv')

    submission_path = kfold_dir / submission_name

    submission_df.to_csv(submission_path, index=False)

    print(f"\n✅ 추론 완료!")
    print(f"   예측 개수: {len(predictions)}")
    print(f"   Submission 저장: {submission_path}")

    # 클래스 분포 출력
    print(f"\n📊 예측 클래스 분포:")
    for cls in range(17):
        count = (submission_df['target'] == cls).sum()
        print(f"   Class {cls}: {count}개 ({count/len(predictions)*100:.1f}%)")


if __name__ == "__main__":
    main()
