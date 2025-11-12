"""
Loss 함수 모듈

다양한 Loss 함수를 제공합니다:
- FocalLoss: 어려운 샘플에 자동 집중
- AsymmetricLoss: False Positive 감소
- LabelSmoothingCrossEntropy: 과적합 방지

클래스 3-7, 14번 같은 어려운 클래스 대응에 효과적입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) - 어려운 샘플에 자동 집중!

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where:
        - p_t: 정답 클래스의 예측 확률
        - alpha: 클래스 균형 파라미터 (보통 0.25)
        - gamma: focusing 파라미터 (보통 2.0)

    gamma가 클수록 어려운 샘플(낮은 확률)에 더 집중합니다:
    - gamma=0: Cross Entropy와 동일
    - gamma=2: 확률 0.9인 샘플의 loss가 0.01배로 감소
    - gamma=5: 확률 0.9인 샘플의 loss가 0.00001배로 감소

    클래스 3-7, 14처럼 어려운 클래스에 자동으로 더 많은 가중치를 부여합니다!

    Args:
        alpha: 클래스 균형 파라미터 (0~1, 기본값: 0.25)
        gamma: Focusing 파라미터 (0~5, 기본값: 2.0)
        reduction: 'mean' 또는 'sum'
        label_smoothing: Label smoothing 비율 (0~1, 기본값: 0.0)
        weight: 클래스별 가중치 (Optional[Tensor])

    예시:
        >>> # 클래스 3-7, 14에 집중하려면 gamma를 높게
        >>> criterion = FocalLoss(alpha=0.25, gamma=3.0, label_smoothing=0.1)
        >>> loss = criterion(outputs, labels)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 모델 출력 logits (shape: [batch_size, num_classes])
            targets: 정답 레이블 (shape: [batch_size])

        Returns:
            Focal loss (scalar)
        """
        # Cross Entropy Loss with Label Smoothing
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
            weight=self.weight,
        )

        # p_t 계산 (정답 클래스의 확률)
        p = torch.exp(-ce_loss)  # p = exp(-ce_loss)

        # Focal Loss 계산
        # (1 - p)^gamma 항이 어려운 샘플에 높은 가중치를 부여
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., 2021) - Multi-label classification에 효과적

    Positive와 Negative 샘플에 다른 gamma 적용:
    - Positive: 어려운 샘플 (gamma_pos)
    - Negative: 쉬운 샘플 (gamma_neg)

    이미지 분류에서 False Positive를 줄이는 데 효과적입니다.

    Args:
        gamma_pos: Positive 샘플의 focusing 파라미터 (기본값: 0)
        gamma_neg: Negative 샘플의 focusing 파라미터 (기본값: 4)
        clip: Negative probability clipping (기본값: 0.05)

    예시:
        >>> criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
        >>> loss = criterion(outputs, labels)
    """

    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 모델 출력 logits (shape: [batch_size, num_classes])
            targets: 정답 레이블 (shape: [batch_size])

        Returns:
            Asymmetric loss (scalar)
        """
        # One-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Sigmoid 확률
        probs = torch.sigmoid(inputs)

        # Clip negative probabilities
        probs = torch.clamp(probs, min=self.clip)

        # Positive loss (Focal Loss)
        pos_loss = -targets_one_hot * (1 - probs) ** self.gamma_pos * torch.log(probs)

        # Negative loss (Focal Loss)
        neg_loss = -(1 - targets_one_hot) * probs ** self.gamma_neg * torch.log(1 - probs)

        # Total loss
        loss = pos_loss + neg_loss
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss

    One-hot encoding 대신 smooth label 사용:
    - 정답 클래스: 1 - smoothing
    - 나머지 클래스: smoothing / (num_classes - 1)

    과적합을 방지하고 모델의 일반화 성능을 향상시킵니다.

    Args:
        smoothing: Label smoothing 비율 (0~1, 기본값: 0.1)
        reduction: 'mean' 또는 'sum'
        weight: 클래스별 가중치 (Optional[Tensor])

    예시:
        >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> loss = criterion(outputs, labels)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 모델 출력 logits (shape: [batch_size, num_classes])
            targets: 정답 레이블 (shape: [batch_size])

        Returns:
            Label smoothing cross entropy loss (scalar)
        """
        num_classes = inputs.size(1)

        # Log softmax
        log_probs = F.log_softmax(inputs, dim=1)

        # Smooth labels
        with torch.no_grad():
            # One-hot encoding
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Loss 계산
        loss = -torch.sum(true_dist * log_probs, dim=1)

        # Class weight 적용
        if self.weight is not None:
            weight = self.weight[targets]
            loss = loss * weight

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_from_config(cfg, device: str = 'cuda') -> nn.Module:
    """
    Hydra config로부터 Loss 함수 생성

    Args:
        cfg: Hydra config (cfg.loss 섹션)
        device: 디바이스 ('cuda' 또는 'cpu')

    Returns:
        Loss 함수 인스턴스

    예시:
        >>> @hydra.main(config_path="configs", config_name="config")
        >>> def main(cfg):
        >>>     criterion = create_loss_from_config(cfg)
    """
    loss_cfg = cfg.loss
    loss_type = loss_cfg.get('type', 'cross_entropy')

    print(f"📊 Loss 함수 생성: {loss_type}")

    if loss_type == 'focal':
        # Focal Loss
        alpha = loss_cfg.get('alpha', 0.25)
        gamma = loss_cfg.get('gamma', 2.0)
        label_smoothing = loss_cfg.get('label_smoothing', 0.0)

        # Class weights
        class_weights = loss_cfg.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float().to(device)

        criterion = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
            weight=class_weights,
        )

        print(f"   Alpha: {alpha}")
        print(f"   Gamma: {gamma}")
        if label_smoothing > 0:
            print(f"   Label Smoothing: {label_smoothing}")
        if class_weights is not None:
            print(f"   Class Weights: {class_weights.cpu().tolist()}")

    elif loss_type == 'asymmetric':
        # Asymmetric Loss
        gamma_pos = loss_cfg.get('gamma_pos', 0)
        gamma_neg = loss_cfg.get('gamma_neg', 4)
        clip = loss_cfg.get('clip', 0.05)

        criterion = AsymmetricLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )

        print(f"   Gamma Pos: {gamma_pos}")
        print(f"   Gamma Neg: {gamma_neg}")
        print(f"   Clip: {clip}")

    elif loss_type == 'label_smoothing':
        # Label Smoothing CE
        smoothing = loss_cfg.get('label_smoothing', 0.1)

        # Class weights
        class_weights = loss_cfg.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float().to(device)

        criterion = LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            weight=class_weights,
        )

        print(f"   Smoothing: {smoothing}")
        if class_weights is not None:
            print(f"   Class Weights: {class_weights.cpu().tolist()}")

    elif loss_type == 'weighted':
        # Weighted Cross Entropy
        class_weights = loss_cfg.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float().to(device)

        label_smoothing = loss_cfg.get('label_smoothing', 0.0)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        print(f"   Class Weights: {class_weights.cpu().tolist() if class_weights is not None else 'None'}")
        if label_smoothing > 0:
            print(f"   Label Smoothing: {label_smoothing}")

    else:
        # 일반 Cross Entropy
        label_smoothing = loss_cfg.get('label_smoothing', 0.0)

        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        if label_smoothing > 0:
            print(f"   Label Smoothing: {label_smoothing}")

    print(f"\n💡 팁: Focal Loss는 클래스 3-7, 14 같은 어려운 샘플에 자동 집중!")
    print(f"   gamma=2.0 (보통) → gamma=3.0 (강하게) → gamma=5.0 (매우 강하게)")

    return criterion
