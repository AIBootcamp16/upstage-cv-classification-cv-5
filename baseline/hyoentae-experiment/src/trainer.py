"""
Trainer 모듈

이 모듈은 모델 학습을 위한 Trainer 클래스를 제공합니다.
학습, 검증, Early Stopping, MixUp/CutMix 등을 지원합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Callable

from src.utils.metrics import (
    calculate_macro_f1,
    calculate_class_f1,
    accuracy_from_logits,
    MetricTracker
)
from src.utils.logger import WandBLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.mixup import mixup_criterion


class Trainer:
    """
    모델 학습을 위한 Trainer 클래스

    주요 기능:
    - Train/Validation 루프
    - MixUp/CutMix 지원
    - WandB 로깅
    - Early Stopping
    - Best 모델 저장
    - Mixed Precision Training (AMP)

    예시:
        >>> trainer = Trainer(
        >>>     model=model,
        >>>     train_loader=train_loader,
        >>>     val_loader=val_loader,
        >>>     criterion=criterion,
        >>>     optimizer=optimizer,
        >>>     device='cuda',
        >>>     logger=wandb_logger,
        >>>     checkpoint_manager=ckpt_manager
        >>> )
        >>> trainer.train(num_epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[WandBLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        mixup_cutmix: Optional[Callable] = None,
        use_amp: bool = False,
        model_config: Optional[dict] = None,
    ):
        """
        Trainer 초기화

        Args:
            model: PyTorch 모델
            train_loader: Train DataLoader
            val_loader: Validation DataLoader
            criterion: Loss 함수
            optimizer: Optimizer
            device: 디바이스 ('cuda' 또는 'cpu')
            scheduler: Learning rate scheduler (옵션)
            logger: WandB 로거 (옵션)
            checkpoint_manager: 체크포인트 매니저 (옵션)
            mixup_cutmix: MixUp/CutMix 증강 함수 (옵션)
            model_config: 모델 config (추론 시 재현용)
            use_amp: Mixed Precision Training 사용 여부
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.mixup_cutmix = mixup_cutmix
        self.use_amp = use_amp
        self.model_config = model_config  # 추론 시 재현용

        # Mixed Precision Training용 GradScaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed Precision Training (AMP) 활성화")

        # Metric Tracker
        self.metric_tracker = MetricTracker()

        print("✅ Trainer 초기화 완료")

    def train_epoch(self, epoch: int) -> dict:
        """
        1 Epoch 학습

        Args:
            epoch: 현재 epoch 번호

        Returns:
            학습 메트릭 딕셔너리
            {'loss': float, 'accuracy': float, 'macro_f1': float, 'lr': float}
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Macro F1 계산용
        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 원본 labels 저장 (Macro F1 계산용)
            original_labels = labels.clone()

            # MixUp/CutMix 적용 (있으면)
            if self.mixup_cutmix is not None:
                images, labels_a, labels_b, lam = self.mixup_cutmix(images, labels)

            # Forward pass
            if self.use_amp:
                # Mixed Precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)

                    # Loss 계산
                    if self.mixup_cutmix is not None:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
            else:
                # 일반 precision
                outputs = self.model(images)

                if self.mixup_cutmix is not None:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 메트릭 계산
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Predictions와 labels 저장 (Macro F1용)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(original_labels.cpu().numpy())

            # Accuracy
            if self.mixup_cutmix is None:
                correct = (predictions == labels).sum().item()
                total_correct += correct

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / total_samples,
            })

        # NumPy 배열로 변환
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Epoch 평균 메트릭
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples if self.mixup_cutmix is None else 0.0

        # Train Macro F1 계산
        train_macro_f1 = calculate_macro_f1(all_predictions, all_labels)

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'train_macro_f1': train_macro_f1,
            'lr': current_lr,
        }

        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict:
        """
        1 Epoch 검증

        Args:
            epoch: 현재 epoch 번호

        Returns:
            검증 메트릭 딕셔너리
            {'loss': float, 'accuracy': float, 'macro_f1': float}
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_labels = []

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 메트릭 누적
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Prediction
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Progress bar 업데이트
            pbar.set_postfix({'loss': loss.item()})

        # NumPy 배열로 변환
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # 메트릭 계산
        avg_loss = total_loss / total_samples
        accuracy = (all_predictions == all_labels).mean()
        macro_f1 = calculate_macro_f1(all_predictions, all_labels)

        # 클래스별 F1 (로깅용)
        class_f1 = calculate_class_f1(all_predictions, all_labels, num_classes=17)

        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_macro_f1': macro_f1,
            'class_f1': class_f1,
        }

        return metrics

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        전체 학습 루프

        Args:
            num_epochs: 총 epoch 수
            start_epoch: 시작 epoch (체크포인트 재개 시 사용)
        """
        print(f"\n🚀 학습 시작: {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print()

        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(epoch + 1)

            # Validation
            val_metrics = self.validate_epoch(epoch + 1)

            # Scheduler step (있으면)
            if self.scheduler is not None:
                self.scheduler.step()

            # Train-Val 차이 계산 (과적합 모니터링)
            loss_gap = train_metrics['train_loss'] - val_metrics['val_loss']
            f1_gap = val_metrics['val_macro_f1'] - train_metrics['train_macro_f1']

            # 메트릭 출력
            print(f"\n📊 Epoch {epoch+1} 결과:")
            print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"   Train Macro F1: {train_metrics['train_macro_f1']:.4f}")
            print(f"   Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"   Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"   Val Macro F1: {val_metrics['val_macro_f1']:.4f}")
            print(f"   Learning Rate: {train_metrics['lr']:.6f}")

            # Train-Val 차이 출력 (과적합 감지)
            print(f"\n🔍 Train-Val 차이 (과적합 모니터링):")
            print(f"   Loss Gap: {loss_gap:+.4f} (Train - Val)")
            print(f"   F1 Gap: {f1_gap:+.4f} (Val - Train)")

            # 과적합 경고
            if f1_gap < -0.05:  # Train F1이 Val F1보다 5%p 이상 높으면
                print(f"   ⚠️  과적합 주의! Train F1이 Val F1보다 {abs(f1_gap):.1%} 높습니다")
            elif f1_gap > 0.05:  # Val F1이 Train F1보다 5%p 이상 높으면
                print(f"   ⚠️  과소적합 주의! Val F1이 Train F1보다 {f1_gap:.1%} 높습니다")
            else:
                print(f"   ✅ 적절한 학습 상태 (차이: {abs(f1_gap):.1%})")

            # WandB 로깅
            if self.logger is not None:
                # 기본 메트릭
                log_dict = {**train_metrics, **val_metrics}
                # 클래스별 F1 제외 (너무 많아서)
                log_dict.pop('class_f1', None)

                # Train-Val 차이 추가
                log_dict['train_val_loss_gap'] = loss_gap
                log_dict['train_val_f1_gap'] = f1_gap

                self.logger.log(log_dict, step=epoch + 1)

                # 클래스별 F1 (별도 로깅)
                self.logger.log_class_metrics(val_metrics['class_f1'], step=epoch + 1)

            # Metric Tracker 업데이트
            self.metric_tracker.update(val_metrics['val_macro_f1'], epoch=epoch + 1)

            # Checkpoint 저장
            if self.checkpoint_manager is not None:
                extra_info = {
                    'val_loss': val_metrics['val_loss'],
                    'val_accuracy': val_metrics['val_accuracy'],
                }

                # 모델 config 추가 (추론 시 자동 모델 로드용)
                if self.model_config is not None:
                    extra_info['model_config'] = self.model_config

                is_best = self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metric_value=val_metrics['val_macro_f1'],
                    train_metric_value=train_metrics['train_macro_f1'],  # Train 메트릭 추가
                    scheduler=self.scheduler,
                    extra_info=extra_info
                )

                # Early Stopping 체크
                if self.checkpoint_manager.should_stop():
                    print(f"\n⚠️  Early Stopping 발동! (Patience={self.checkpoint_manager.patience})")
                    print(f"   Best Macro F1: {self.checkpoint_manager.get_best_metric():.4f}")
                    print(f"   Best Epoch: {self.checkpoint_manager.get_best_epoch()}")
                    break

        print(f"\n✅ 학습 완료!")
        if self.checkpoint_manager is not None:
            print(f"   Best Macro F1: {self.checkpoint_manager.get_best_metric():.4f}")
            print(f"   Best Epoch: {self.checkpoint_manager.get_best_epoch()}")


def create_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Config로부터 Optimizer 생성

    Args:
        model: PyTorch 모델
        cfg: Hydra config (cfg.train.optimizer 섹션)

    Returns:
        Optimizer

    예시:
        >>> optimizer = create_optimizer(model, cfg)
    """
    optimizer_cfg = cfg.train.optimizer

    optimizer_name = optimizer_cfg.get('name', 'AdamW').lower()
    lr = optimizer_cfg.get('lr', 0.001)
    weight_decay = optimizer_cfg.get('weight_decay', 0.01)

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        momentum = optimizer_cfg.get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"✅ Optimizer 생성: {optimizer_name}")
    print(f"   Learning Rate: {lr}")
    print(f"   Weight Decay: {weight_decay}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, cfg, num_epochs: int):
    """
    Config로부터 Learning Rate Scheduler 생성

    Args:
        optimizer: Optimizer
        cfg: Hydra config (cfg.train.scheduler 섹션)
        num_epochs: 총 epoch 수

    Returns:
        Scheduler 또는 None

    예시:
        >>> scheduler = create_scheduler(optimizer, cfg, num_epochs=50)
    """
    scheduler_cfg = cfg.train.get('scheduler', None)

    if scheduler_cfg is None:
        return None

    scheduler_name = scheduler_cfg.get('name', 'cosine').lower()

    if scheduler_name == 'cosine':
        # Cosine Annealing with Warmup
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
        min_lr = scheduler_cfg.get('min_lr', 1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )

        print(f"✅ Scheduler 생성: CosineAnnealing")
        print(f"   Warmup Epochs: {warmup_epochs}")
        print(f"   Min LR: {min_lr}")

    elif scheduler_name == 'step':
        step_size = scheduler_cfg.get('step_size', 10)
        gamma = scheduler_cfg.get('gamma', 0.1)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

        print(f"✅ Scheduler 생성: StepLR")
        print(f"   Step Size: {step_size}")
        print(f"   Gamma: {gamma}")

    else:
        print(f"⚠️  Unknown scheduler: {scheduler_name}, 스케줄러 없이 진행")
        return None

    return scheduler
