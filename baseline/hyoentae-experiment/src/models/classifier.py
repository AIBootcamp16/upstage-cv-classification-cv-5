"""
모델 분류기 모듈

이 모듈은 timm 라이브러리를 사용한 이미지 분류 모델을 제공합니다.

지원 모델:
- EfficientNet (B0, B3)
- ResNet50
- ResNeXt50
- ConvNeXt Tiny
"""

import timm
import torch
import torch.nn as nn
from typing import Optional


class DocumentClassifier(nn.Module):
    """
    문서 분류를 위한 모델 래퍼

    timm 라이브러리의 pre-trained 모델을 사용하여
    17개 클래스 문서 분류를 수행합니다.

    예시:
        >>> model = DocumentClassifier(
        >>>     model_name='efficientnet_b0',
        >>>     num_classes=17,
        >>>     pretrained=True,
        >>>     dropout=0.2
        >>> )
        >>> outputs = model(images)  # [batch_size, 17]
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 17,
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: Optional[float] = None,
    ):
        """
        모델 초기화

        Args:
            model_name: timm 모델 이름
                       예: 'efficientnet_b0', 'resnet50', 'convnext_tiny'
            num_classes: 출력 클래스 수 (기본값: 17)
            pretrained: ImageNet pre-trained 가중치 사용 여부
            dropout: Dropout 비율 (과적합 방지)
            drop_path_rate: DropPath 비율 (ConvNeXt 등에서 사용)
        """
        super(DocumentClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # timm 모델 생성
        if drop_path_rate is not None:
            # ConvNeXt 등 DropPath를 지원하는 모델
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
                drop_path_rate=drop_path_rate,
            )
        else:
            # 일반 모델
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )

        print(f"✅ 모델 생성 완료: {model_name}")
        print(f"   클래스 수: {num_classes}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Dropout: {dropout}")
        if drop_path_rate is not None:
            print(f"   DropPath: {drop_path_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 입력 이미지 (shape: [batch_size, 3, H, W])

        Returns:
            출력 logits (shape: [batch_size, num_classes])
        """
        return self.backbone(x)

    def get_num_parameters(self) -> int:
        """
        모델의 총 파라미터 수 반환

        Returns:
            파라미터 개수
        """
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """
        학습 가능한 파라미터 수 반환

        Returns:
            학습 가능한 파라미터 개수
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(cfg, num_classes: int = 17) -> DocumentClassifier:
    """
    Hydra config로부터 모델 생성

    configs/model/ 폴더의 YAML 설정을 읽어서 모델을 생성합니다.

    Args:
        cfg: Hydra config (cfg.model 섹션 필요)
        num_classes: 클래스 수

    Returns:
        DocumentClassifier 인스턴스

    예시:
        >>> @hydra.main(config_path="configs", config_name="config")
        >>> def main(cfg):
        >>>     model = create_model_from_config(cfg, num_classes=17)
    """
    model_cfg = cfg.model

    # 모델 파라미터 추출
    model_name = model_cfg.get('name')
    pretrained = model_cfg.get('pretrained', True)
    dropout = model_cfg.get('dropout', 0.2)
    drop_path_rate = model_cfg.get('drop_path_rate', None)

    # 모델 생성
    model = DocumentClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )

    # 파라미터 정보 출력
    total_params = model.get_num_parameters()
    trainable_params = model.get_trainable_parameters()
    print(f"📊 모델 파라미터:")
    print(f"   전체: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")

    return model


def get_model_input_size(model_name: str) -> int:
    """
    모델의 기본 입력 크기 반환

    Args:
        model_name: timm 모델 이름

    Returns:
        기본 입력 이미지 크기 (정사각형)

    예시:
        >>> img_size = get_model_input_size('efficientnet_b0')
        >>> print(img_size)  # 224
    """
    # timm의 기본 입력 크기
    default_sizes = {
        'efficientnet_b0': 224,
        'efficientnet_b3': 300,
        'resnet50': 224,
        'resnext50_32x4d': 224,
        'convnext_tiny': 224,
    }

    return default_sizes.get(model_name, 224)


class ModelEMA:
    """
    Exponential Moving Average (EMA) 모델

    학습 중 모델 파라미터의 이동 평균을 유지하여
    더 안정적인 예측을 가능하게 합니다.

    예시:
        >>> model = create_model(cfg)
        >>> ema = ModelEMA(model, decay=0.9999)
        >>> # 학습 루프
        >>> for batch in train_loader:
        >>>     loss = train_step(batch)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update(model)  # EMA 업데이트
        >>> # 평가 시
        >>> ema.apply_shadow()  # EMA 파라미터로 교체
        >>> evaluate(model)
        >>> ema.restore()  # 원래 파라미터로 복원
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        EMA 초기화

        Args:
            model: PyTorch 모델
            decay: EMA decay 비율 (0.999 ~ 0.9999 권장)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 초기 shadow 파라미터 생성
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """
        EMA 파라미터 업데이트

        Args:
            model: 현재 학습 중인 모델
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        모델 파라미터를 EMA 파라미터로 교체
        (평가 전에 호출)
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        모델 파라미터를 원래 값으로 복원
        (평가 후에 호출)
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
