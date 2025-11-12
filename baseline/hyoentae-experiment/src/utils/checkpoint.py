"""
체크포인트 관리 모듈

이 모듈은 모델 체크포인트의 저장과 로드를 관리합니다.
Best 모델 저장, Early Stopping 등의 기능을 제공합니다.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    모델 체크포인트를 관리하는 클래스

    Best 모델 저장, 주기적 체크포인트 저장, Early Stopping 등을 처리합니다.

    예시:
        >>> manager = CheckpointManager(save_dir="outputs", patience=10)
        >>> manager.save_checkpoint(model, optimizer, epoch, val_f1)
        >>> if manager.should_stop():
        >>>     print("Early stopping!")
    """

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "macro_f1",
        mode: str = "max",
        patience: int = 10,
        verbose: bool = True,
        use_generalization_score: bool = True,
        overfitting_penalty: float = 0.3,
    ):
        """
        체크포인트 매니저 초기화

        Args:
            save_dir: 체크포인트 저장 디렉토리
            metric_name: 추적할 메트릭 이름 (예: "macro_f1")
            mode: 메트릭 모드 ("max" 또는 "min")
                 - "max": 값이 클수록 좋음 (accuracy, f1 등)
                 - "min": 값이 작을수록 좋음 (loss 등)
            patience: Early stopping patience (몇 epoch 동안 개선 없으면 중단)
            verbose: 로그 출력 여부
            use_generalization_score: Generalization score 사용 여부
                 - True: Val 메트릭 - 과적합 페널티로 best 판단
                 - False: Val 메트릭만으로 best 판단
            overfitting_penalty: 과적합 페널티 가중치 (0.0~1.0)
                 - 0.0: 과적합 무시 (Val만 고려)
                 - 0.3 (권장): 적당한 과적합 방지
                 - 0.5: 강한 과적합 방지
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.use_generalization_score = use_generalization_score
        self.overfitting_penalty = overfitting_penalty

        # Best 메트릭 초기화
        if mode == "max":
            self.best_metric = float('-inf')
            self.best_score = float('-inf')  # Generalization score
            self.compare = lambda x, y: x > y
        else:  # mode == "min"
            self.best_metric = float('inf')
            self.best_score = float('inf')
            self.compare = lambda x, y: x < y

        # Early stopping 카운터
        self.counter = 0
        self.best_epoch = 0
        self.best_train_metric = None  # Best epoch의 Train 메트릭

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        train_metric_value: Optional[float] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        체크포인트 저장

        메트릭이 개선되었으면 best.pth로 저장하고 True 반환,
        아니면 False 반환합니다.

        Args:
            model: PyTorch 모델
            optimizer: Optimizer
            epoch: 현재 epoch
            metric_value: 현재 Val 메트릭 값
            train_metric_value: 현재 Train 메트릭 값 (Generalization score 계산용, 옵션)
            scheduler: Learning rate scheduler (옵션)
            extra_info: 추가 정보 딕셔너리 (옵션)

        Returns:
            Best 모델이 갱신되었으면 True, 아니면 False

        예시:
            >>> is_best = manager.save_checkpoint(
            >>>     model, optimizer, epoch=10,
            >>>     metric_value=0.85, train_metric_value=0.90
            >>> )
            >>> if is_best:
            >>>     print("새로운 best 모델 저장!")
        """
        # Generalization Score 계산
        if self.use_generalization_score and train_metric_value is not None:
            # 과적합 페널티 계산
            if self.mode == "max":
                # F1 등: Train이 Val보다 높으면 과적합
                gap = max(0, train_metric_value - metric_value)
                score = metric_value - self.overfitting_penalty * gap
            else:  # mode == "min"
                # Loss 등: Val이 Train보다 높으면 과적합
                gap = max(0, metric_value - train_metric_value)
                score = metric_value + self.overfitting_penalty * gap

            # Score 기준으로 개선 여부 판단
            is_improved = self.compare(score, self.best_score)
            current_score = score
        else:
            # 기존 방식: Val 메트릭만 사용
            is_improved = self.compare(metric_value, self.best_metric)
            current_score = metric_value

        if is_improved:
            # Best 메트릭 업데이트
            self.best_metric = metric_value
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_train_metric = train_metric_value  # Best epoch의 Train 메트릭 저장
            self.counter = 0  # Early stopping 카운터 리셋

            # 체크포인트 구성
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                self.metric_name: metric_value,
                'best_metric': self.best_metric,
            }

            # Train 메트릭도 저장 (있으면)
            if train_metric_value is not None:
                checkpoint['train_' + self.metric_name] = train_metric_value
                checkpoint['generalization_score'] = current_score

            # Scheduler가 있으면 추가
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            # 추가 정보가 있으면 추가
            if extra_info is not None:
                checkpoint.update(extra_info)

            # Best 모델 저장
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)

            if self.verbose:
                print(f"✅ Best 모델 저장: {self.metric_name}={metric_value:.4f} (Epoch {epoch})")
                if self.use_generalization_score and train_metric_value is not None:
                    gap = train_metric_value - metric_value if self.mode == "max" else metric_value - train_metric_value
                    print(f"   Generalization Score: {current_score:.4f}")
                    print(f"   Train-Val Gap: {gap:+.4f} (Penalty: {self.overfitting_penalty * max(0, gap):.4f})")
                print(f"   저장 경로: {best_path}")

            return True

        else:
            # 메트릭 개선 안 됨 -> Early stopping 카운터 증가
            self.counter += 1

            if self.verbose:
                if self.use_generalization_score and train_metric_value is not None:
                    print(f"⚠️  메트릭 개선 없음: {self.metric_name}={metric_value:.4f}, Score={current_score:.4f} "
                          f"(Best: {self.best_metric:.4f}, Best Score: {self.best_score:.4f}, "
                          f"Patience: {self.counter}/{self.patience})")
                else:
                    print(f"⚠️  메트릭 개선 없음: {self.metric_name}={metric_value:.4f} "
                          f"(Best: {self.best_metric:.4f}, Patience: {self.counter}/{self.patience})")

            return False

    def save_last_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        마지막 체크포인트 저장 (best와 별개)

        학습 재개를 위해 마지막 epoch의 상태를 저장합니다.

        Args:
            model: PyTorch 모델
            optimizer: Optimizer
            epoch: 현재 epoch
            metric_value: 현재 메트릭 값
            scheduler: Learning rate scheduler (옵션)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.metric_name: metric_value,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # last.pth로 저장
        last_path = self.save_dir / "last.pth"
        torch.save(checkpoint, last_path)

        if self.verbose:
            print(f"💾 마지막 체크포인트 저장: {last_path}")

    def should_stop(self) -> bool:
        """
        Early stopping 여부 확인

        Patience만큼 epoch 동안 메트릭 개선이 없으면 True 반환

        Returns:
            Early stopping 해야 하면 True

        예시:
            >>> if manager.should_stop():
            >>>     print("Early stopping 발동!")
            >>>     break
        """
        return self.counter >= self.patience

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        체크포인트 로드

        저장된 체크포인트를 읽어서 모델과 optimizer에 적용합니다.

        Args:
            checkpoint_path: 체크포인트 파일 경로
            model: PyTorch 모델 (state_dict가 로드됨)
            optimizer: Optimizer (옵션, state_dict가 로드됨)
            scheduler: Scheduler (옵션, state_dict가 로드됨)
            device: 디바이스 ("cuda" 또는 "cpu")

        Returns:
            체크포인트 딕셔너리 (epoch, metric 등 포함)

        예시:
            >>> checkpoint = manager.load_checkpoint(
            >>>     "outputs/best.pth", model, optimizer, device="cuda"
            >>> )
            >>> print(f"Loaded epoch: {checkpoint['epoch']}")
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        # 체크포인트 로드 (PyTorch 2.6+ 호환성)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 모델 state_dict 로드
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optimizer state_dict 로드 (있으면)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Scheduler state_dict 로드 (있으면)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            metric_val = checkpoint.get(self.metric_name, 'unknown')
            print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
            print(f"   Epoch: {epoch}, {self.metric_name}: {metric_val}")

        return checkpoint

    def get_best_metric(self) -> float:
        """
        Best 메트릭 값 반환

        Returns:
            Best 메트릭 값
        """
        return self.best_metric

    def get_best_epoch(self) -> int:
        """
        Best 메트릭을 기록한 epoch 반환

        Returns:
            Best epoch 번호
        """
        return self.best_epoch

    def get_best_train_metric(self) -> Optional[float]:
        """
        Best epoch의 Train 메트릭 값 반환

        Returns:
            Best epoch의 Train 메트릭 값 (없으면 None)
        """
        return self.best_train_metric


def find_latest_checkpoint(output_dir: str = "outputs") -> str:
    """
    가장 최근 실험의 best.pth 경로를 자동으로 찾습니다.

    Args:
        output_dir: outputs 디렉토리 경로

    Returns:
        최신 best.pth 경로

    Raises:
        FileNotFoundError: best.pth를 찾을 수 없을 때

    예시:
        >>> checkpoint_path = find_latest_checkpoint(output_dir="outputs")
        >>> print(checkpoint_path)  # outputs/2025-11-02/12-34-56/best.pth
    """
    import glob
    from pathlib import Path

    output_path = Path(output_dir)

    # outputs/**/best.pth 패턴으로 모든 best.pth 찾기
    checkpoint_pattern = str(output_path / "**" / "best.pth")
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)

    if not checkpoints:
        raise FileNotFoundError(
            f"❌ {output_dir}/ 디렉토리에서 best.pth를 찾을 수 없습니다.\n"
            f"   먼저 train.py를 실행해서 모델을 학습시켜주세요."
        )

    # 파일 수정 시간으로 정렬 (가장 최근 것 선택)
    latest_checkpoint = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)

    print(f"🔍 자동 검색: 최신 체크포인트 발견")
    print(f"   경로: {latest_checkpoint}")

    # 해당 실험의 config도 출력
    checkpoint_dir = Path(latest_checkpoint).parent
    hydra_config = checkpoint_dir / ".hydra" / "config.yaml"
    if hydra_config.exists():
        import yaml
        with open(hydra_config, 'r') as f:
            config = yaml.safe_load(f)
            model_name = config.get('model', {}).get('name', 'unknown')
            aug_type = config.get('augmentation', {}).get('name', 'unknown')
            print(f"   모델: {model_name}, 증강: {aug_type}")

    return latest_checkpoint


def load_model_for_inference(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    추론용 모델 로드 (간단 버전)

    체크포인트에서 모델만 로드하고 eval 모드로 전환합니다.

    Args:
        model: PyTorch 모델
        checkpoint_path: 체크포인트 경로
        device: 디바이스

    Returns:
        로드되고 eval 모드로 설정된 모델

    예시:
        >>> model = create_model(cfg)
        >>> model = load_model_for_inference(model, "outputs/best.pth", "cuda")
        >>> predictions = model(inputs)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    # 체크포인트 로드 (PyTorch 2.6+ 호환성)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 모델 state_dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])

    # Eval 모드로 전환
    model.eval()

    # 디바이스로 이동
    model = model.to(device)

    print(f"✅ 추론용 모델 로드 완료: {checkpoint_path}")

    return model
