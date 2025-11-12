"""
MixUp & CutMix 데이터 증강 모듈

이 모듈은 소량의 Train 데이터(1,570개)를 효과적으로 증강하기 위한
MixUp과 CutMix 기법을 제공합니다.

- MixUp: 두 이미지를 alpha 비율로 섞음
- CutMix: 한 이미지의 일부를 다른 이미지로 교체
"""

import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    MixUp 데이터 증강

    두 이미지를 lambda 비율로 섞어서 새로운 학습 샘플을 생성합니다.
    Label도 같은 비율로 섞어서 soft label을 만듭니다.

    공식:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    Args:
        x: 입력 이미지 배치 (shape: [batch_size, C, H, W])
        y: 레이블 배치 (shape: [batch_size])
        alpha: Beta 분포 파라미터 (높을수록 더 많이 섞임)

    Returns:
        mixed_x: 섞인 이미지 (shape: [batch_size, C, H, W])
        y_a: 첫 번째 레이블 (shape: [batch_size])
        y_b: 두 번째 레이블 (shape: [batch_size])
        lam: 섞음 비율 (0~1)

    예시:
        >>> mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
        >>> # Loss 계산 시
        >>> loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
    """
    if alpha > 0:
        # Beta 분포에서 lambda 샘플링
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # 랜덤하게 배치 내 이미지 순서 섞기
    index = torch.randperm(batch_size).to(x.device)

    # 이미지 섞기
    mixed_x = lam * x + (1 - lam) * x[index]

    # 레이블 2개 반환 (나중에 loss 계산 시 사용)
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp용 Loss 계산

    두 레이블에 대한 loss를 lambda 비율로 섞습니다.

    Args:
        criterion: Loss 함수 (예: nn.CrossEntropyLoss())
        pred: 모델 예측 (shape: [batch_size, num_classes])
        y_a: 첫 번째 레이블
        y_b: 두 번째 레이블
        lam: 섞음 비율

    Returns:
        MixUp loss

    예시:
        >>> criterion = nn.CrossEntropyLoss()
        >>> mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
        >>> preds = model(mixed_x)
        >>> loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    CutMix용 랜덤 Bounding Box 생성

    이미지에서 잘라낼 영역의 좌표를 생성합니다.
    영역 크기는 lambda에 비례합니다.

    Args:
        size: 이미지 크기 (W, H)
        lam: 섞음 비율 (영역 크기 결정)

    Returns:
        (bbx1, bby1, bbx2, bby2): Bounding box 좌표
    """
    W = size[2]
    H = size[3]

    # 잘라낼 영역의 크기 계산
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 중심점 랜덤 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box 좌표 (이미지 경계 안에 있도록 clip)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    CutMix 데이터 증강

    한 이미지의 일부 영역을 다른 이미지의 영역으로 교체합니다.
    문서/자동차 이미지를 섞어서 robust한 학습이 가능합니다.

    Args:
        x: 입력 이미지 배치 (shape: [batch_size, C, H, W])
        y: 레이블 배치 (shape: [batch_size])
        alpha: Beta 분포 파라미터

    Returns:
        cutmix_x: CutMix 적용된 이미지
        y_a: 첫 번째 레이블
        y_b: 두 번째 레이블
        lam: 섞음 비율 (실제 픽셀 비율로 조정됨)

    예시:
        >>> cutmix_x, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
        >>> preds = model(cutmix_x)
        >>> loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # 랜덤하게 배치 내 이미지 순서 섞기
    index = torch.randperm(batch_size).to(x.device)

    # 잘라낼 영역 생성
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # CutMix 적용 (일부 영역만 교체)
    cutmix_x = x.clone()
    cutmix_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Lambda를 실제 픽셀 비율로 조정
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a = y
    y_b = y[index]

    return cutmix_x, y_a, y_b, lam


class MixUpCutMixAugmentation:
    """
    MixUp과 CutMix를 랜덤하게 적용하는 클래스

    학습 시 batch마다 MixUp 또는 CutMix를 랜덤하게 선택해서 적용합니다.

    예시:
        >>> augmentation = MixUpCutMixAugmentation(
        >>>     mixup_alpha=0.2,
        >>>     cutmix_alpha=1.0,
        >>>     mixup_prob=0.5,
        >>>     cutmix_prob=0.5
        >>> )
        >>> mixed_x, y_a, y_b, lam = augmentation(images, labels)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
    ):
        """
        MixUp/CutMix 증강 초기화

        Args:
            mixup_alpha: MixUp Beta 분포 파라미터
            cutmix_alpha: CutMix Beta 분포 파라미터
            mixup_prob: MixUp 적용 확률 (0~1)
            cutmix_prob: CutMix 적용 확률 (0~1)
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        MixUp 또는 CutMix 랜덤 적용

        Args:
            x: 입력 이미지 배치
            y: 레이블 배치

        Returns:
            augmented_x, y_a, y_b, lam
        """
        # 랜덤하게 MixUp 또는 CutMix 선택
        r = np.random.rand()

        if r < self.mixup_prob / (self.mixup_prob + self.cutmix_prob):
            # MixUp 적용
            return mixup_data(x, y, alpha=self.mixup_alpha)
        elif r < (self.mixup_prob + self.cutmix_prob):
            # CutMix 적용
            return cutmix_data(x, y, alpha=self.cutmix_alpha)
        else:
            # 둘 다 적용 안 함
            return x, y, y, 1.0


def create_mixup_cutmix_from_config(cfg):
    """
    Config로부터 MixUp/CutMix 증강 생성

    Hydra config의 train 섹션을 읽어서 증강 객체를 생성합니다.

    Args:
        cfg: Hydra config (cfg.train.mixup_alpha, cfg.train.cutmix_alpha)

    Returns:
        MixUpCutMixAugmentation 객체 또는 None

    예시:
        >>> augmentation = create_mixup_cutmix_from_config(cfg)
        >>> if augmentation is not None:
        >>>     mixed_x, y_a, y_b, lam = augmentation(images, labels)
    """
    # cfg.train에서 직접 읽기 (SOTA 설정 방식)
    train_cfg = cfg.get('train', {})

    mixup_alpha = train_cfg.get('mixup_alpha', 0.0)
    cutmix_alpha = train_cfg.get('cutmix_alpha', 0.0)

    # 둘 다 0이면 None 반환
    if mixup_alpha == 0.0 and cutmix_alpha == 0.0:
        return None

    # MixUp/CutMix 객체 생성
    # mixup_prob, cutmix_prob는 alpha 값이 있으면 자동으로 활성화
    augmentation = MixUpCutMixAugmentation(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mixup_prob=1.0 if mixup_alpha > 0 else 0.0,  # alpha > 0이면 항상 적용
        cutmix_prob=0.0,  # CutMix는 비활성화 (SOTA는 MixUp만 사용)
    )

    return augmentation
