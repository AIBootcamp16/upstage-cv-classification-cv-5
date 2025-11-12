"""
Evaluation Script

Validation set에서 모델 성능 평가

사용법:
    # Validation set 평가
    python evaluate.py checkpoint=outputs/2025-11-02/12-00-00/best.pth

    # Config 오버라이드
    python evaluate.py checkpoint=best.pth model=efficientnet_b0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.classifier import create_model_from_config
from src.data.dataset import create_train_val_datasets, create_dataloaders
from src.data.transforms import create_transforms_from_config
from src.utils.checkpoint import load_model_for_inference, find_latest_checkpoint
from src.utils.metrics import (
    calculate_macro_f1,
    calculate_class_f1,
    get_classification_report,
    get_confusion_matrix
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    num_classes: int = 17
) -> dict:
    """
    모델 평가

    Args:
        model: PyTorch 모델
        val_loader: Validation DataLoader
        device: 디바이스
        num_classes: 클래스 수

    Returns:
        평가 결과 딕셔너리
    """
    model.eval()

    all_predictions = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        # 예측
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # NumPy 배열로 변환
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 메트릭 계산
    accuracy = (all_predictions == all_labels).mean()
    macro_f1 = calculate_macro_f1(all_predictions, all_labels)
    class_f1 = calculate_class_f1(all_predictions, all_labels, num_classes=num_classes)

    # Classification Report
    classification_rep = get_classification_report(
        all_predictions, all_labels, num_classes=num_classes
    )

    # Confusion Matrix
    confusion_mat = get_confusion_matrix(
        all_predictions, all_labels, num_classes=num_classes
    )

    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_f1': class_f1,
        'classification_report': classification_rep,
        'confusion_matrix': confusion_mat,
        'predictions': all_predictions,
        'labels': all_labels,
    }

    return results


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: str = "confusion_matrix.png",
    num_classes: int = 17
):
    """
    Confusion Matrix 시각화

    Args:
        confusion_matrix: Confusion matrix 배열
        save_path: 저장 경로
        num_classes: 클래스 수
    """
    plt.figure(figsize=(14, 12))

    # 클래스 이름
    class_names = [f'target_{i}' for i in range(num_classes)]

    # Heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion Matrix saved: {save_path}")

    plt.close()


def plot_class_f1_scores(
    class_f1: dict,
    save_path: str = "class_f1_scores.png"
):
    """
    클래스별 F1 Score 막대 그래프

    Args:
        class_f1: 클래스별 F1 딕셔너리
        save_path: 저장 경로
    """
    plt.figure(figsize=(12, 6))

    # 데이터 준비
    classes = list(class_f1.keys())
    f1_scores = list(class_f1.values())

    # 색상 (F1이 낮은 클래스는 빨간색)
    colors = ['red' if f1 < 0.7 else 'green' if f1 > 0.9 else 'orange' for f1 in f1_scores]

    # 막대 그래프
    bars = plt.bar(classes, f1_scores, color=colors, alpha=0.7, edgecolor='black')

    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.title('F1 Score by Class', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='F1=0.8')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ F1 Score by Class plot saved: {save_path}")

    plt.close()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    메인 평가 함수

    Args:
        cfg: Hydra config
    """
    # Config 출력
    print("\n" + "="*60)
    print("⚙️  Evaluation Config")
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
    train_csv = data_dir / cfg.data.train_csv
    train_img_dir = data_dir / cfg.data.train_dir

    print(f"\n📂 데이터 경로:")
    print(f"   CSV: {train_csv}")
    print(f"   Images: {train_img_dir}\n")

    # Transforms
    train_transform = create_transforms_from_config(cfg, mode='train')
    val_transform = create_transforms_from_config(cfg, mode='valid')

    # Dataset (Validation만 사용)
    _, val_dataset = create_train_val_datasets(
        csv_path=str(train_csv),
        img_dir=str(train_img_dir),
        train_transform=train_transform,
        val_transform=val_transform,
        val_split=cfg.data.val_split,
        random_state=cfg.get('seed', 42),
    )

    # DataLoader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.get('batch_size', 32),
        shuffle=False,
        num_workers=cfg.data.get('num_workers', 4),
        pin_memory=True,
    )

    # 체크포인트에서 모델 설정 읽기
    print(f"\n📦 체크포인트 로드 중: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 체크포인트에서 model config 복원
    if 'model_config' in checkpoint:
        model_cfg = checkpoint['model_config']
        print(f"✅ 체크포인트에서 모델 설정 복원: {model_cfg['architecture']}")

        from src.models.classifier import DocumentClassifier

        model = DocumentClassifier(
            model_name=model_cfg['architecture'],
            num_classes=model_cfg['num_classes'],
            pretrained=False,
            dropout=model_cfg.get('dropout', 0.3)
        )

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"✅ 모델 로드 완료: {model_cfg['architecture']}")

    elif 'config' in checkpoint:
        model_cfg = checkpoint['config']['model']
        print(f"✅ 체크포인트에서 모델 설정 복원: {model_cfg['architecture']}")

        from src.models.classifier import DocumentClassifier

        model = DocumentClassifier(
            model_name=model_cfg['architecture'],
            num_classes=model_cfg['num_classes'],
            pretrained=False,
            dropout=model_cfg.get('dropout', 0.3)
        )

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"✅ 모델 로드 완료: {model_cfg['architecture']}")

    else:
        # 체크포인트에 config가 없으면 현재 config 사용
        print("⚠️  체크포인트에 모델 설정이 없어 현재 config 사용")
        model = create_model_from_config(cfg, num_classes=cfg.data.num_classes)
        model = load_model_for_inference(model, checkpoint_path, device)

    # 평가
    print("\n🔍 모델 평가 시작...\n")
    results = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=cfg.data.num_classes
    )

    # 결과 출력
    print("\n" + "="*60)
    print("📊 평가 결과")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    print("="*60)

    print("\n📋 클래스별 F1 Score:")
    print("-"*40)
    for class_name, f1_score in results['class_f1'].items():
        status = "✅" if f1_score > 0.8 else "⚠️" if f1_score > 0.6 else "❌"
        print(f"  {status} {class_name}: {f1_score:.4f}")

    print("\n📈 Classification Report:")
    print(results['classification_report'])

    # Confusion Matrix 시각화
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path="confusion_matrix.png",
        num_classes=cfg.data.num_classes
    )

    # 클래스별 F1 Score 그래프
    plot_class_f1_scores(
        results['class_f1'],
        save_path="class_f1_scores.png"
    )

    print("\n✅ 평가 완료!")
    print(f"   Macro F1: {results['macro_f1']:.4f}")
    print(f"   Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
