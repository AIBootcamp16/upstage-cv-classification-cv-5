"""
평가 메트릭 계산 모듈

이 모듈은 모델 성능 평가를 위한 메트릭 계산 함수들을 제공합니다.
주요 메트릭: Macro F1 Score (17개 클래스 균등 평가)
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch


def calculate_macro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Macro F1 Score 계산

    17개 클래스 각각의 F1 score를 계산한 후 평균을 냅니다.
    클래스 불균형이 있어도 모든 클래스를 동등하게 평가합니다.

    Args:
        predictions: 예측값 배열 (shape: [N,])
        targets: 실제 레이블 배열 (shape: [N,])

    Returns:
        Macro F1 score (0.0 ~ 1.0)

    예시:
        >>> preds = np.array([0, 1, 2, 0, 1])
        >>> targets = np.array([0, 1, 2, 1, 1])
        >>> macro_f1 = calculate_macro_f1(preds, targets)
        >>> print(f"Macro F1: {macro_f1:.4f}")
    """
    return f1_score(targets, predictions, average='macro')


def calculate_class_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 17) -> dict:
    """
    클래스별 F1 Score 계산

    각 클래스(0-16)별로 F1 score를 계산하여 딕셔너리로 반환합니다.
    어떤 클래스에서 성능이 낮은지 분석할 때 유용합니다.

    Args:
        predictions: 예측값 배열 (shape: [N,])
        targets: 실제 레이블 배열 (shape: [N,])
        num_classes: 클래스 개수 (기본값: 17)

    Returns:
        클래스별 F1 score 딕셔너리
        예: {'class_0': 0.85, 'class_1': 0.92, ...}
    """
    # 각 클래스별 F1 score 계산
    f1_per_class = f1_score(targets, predictions, average=None, labels=range(num_classes))

    # 딕셔너리로 변환
    class_f1_dict = {f'class_{i}': f1_per_class[i] for i in range(num_classes)}

    return class_f1_dict


def get_classification_report(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 17) -> str:
    """
    상세한 분류 리포트 생성

    클래스별 precision, recall, f1-score, support를 포함한
    전체 분류 성능 리포트를 생성합니다.

    Args:
        predictions: 예측값 배열
        targets: 실제 레이블 배열
        num_classes: 클래스 개수

    Returns:
        분류 리포트 문자열
    """
    # 클래스 이름 지정 (target_0 ~ target_16)
    target_names = [f'target_{i}' for i in range(num_classes)]

    return classification_report(
        targets,
        predictions,
        labels=range(num_classes),
        target_names=target_names,
        digits=4
    )


def get_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 17) -> np.ndarray:
    """
    Confusion Matrix 계산

    실제 클래스와 예측 클래스 간의 혼동 행렬을 생성합니다.
    어떤 클래스끼리 헷갈리는지 분석할 때 유용합니다.

    Args:
        predictions: 예측값 배열
        targets: 실제 레이블 배열
        num_classes: 클래스 개수

    Returns:
        Confusion matrix (shape: [num_classes, num_classes])
        행: 실제 클래스, 열: 예측 클래스
    """
    return confusion_matrix(targets, predictions, labels=range(num_classes))


class MetricTracker:
    """
    학습 중 메트릭을 추적하는 클래스

    Epoch마다 메트릭을 저장하고 best score를 추적합니다.
    Early stopping 판단에 사용됩니다.

    예시:
        >>> tracker = MetricTracker()
        >>> tracker.update(0.75)
        >>> tracker.update(0.82)
        >>> print(f"Best: {tracker.best_score}, Current: {tracker.current_score}")
    """

    def __init__(self):
        """메트릭 트래커 초기화"""
        self.scores = []  # 전체 score 기록
        self.best_score = 0.0  # 최고 score
        self.best_epoch = 0  # 최고 score를 기록한 epoch
        self.current_score = 0.0  # 현재 score

    def update(self, score: float, epoch: int = None):
        """
        새로운 score 업데이트

        Args:
            score: 새로운 메트릭 값
            epoch: 현재 epoch (옵션)
        """
        self.scores.append(score)
        self.current_score = score

        # Best score 업데이트
        if score > self.best_score:
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch

    def is_improved(self) -> bool:
        """
        현재 score가 best score인지 확인

        Returns:
            현재 score가 역대 최고면 True
        """
        return self.current_score >= self.best_score

    def get_best(self) -> tuple:
        """
        Best score와 해당 epoch 반환

        Returns:
            (best_score, best_epoch) 튜플
        """
        return self.best_score, self.best_epoch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Logit에서 Accuracy 계산

    모델 출력(logits)과 실제 레이블로부터 정확도를 계산합니다.

    Args:
        logits: 모델 출력 (shape: [batch_size, num_classes])
        targets: 실제 레이블 (shape: [batch_size])

    Returns:
        Accuracy (0.0 ~ 1.0)
    """
    # Logit에서 가장 높은 값의 인덱스 = 예측 클래스
    predictions = torch.argmax(logits, dim=1)

    # 정확도 계산
    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return correct / total
