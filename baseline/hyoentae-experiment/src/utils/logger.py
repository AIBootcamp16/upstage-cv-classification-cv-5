"""
WandB 로깅 모듈

이 모듈은 Weights & Biases를 사용한 실험 추적 기능을 제공합니다.
학습 중 loss, metrics, learning rate 등을 자동으로 기록합니다.
"""

import wandb
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class WandBLogger:
    """
    WandB 로깅을 관리하는 클래스

    학습 과정의 모든 메트릭과 설정을 WandB에 자동으로 기록합니다.

    예시:
        >>> logger = WandBLogger(project="cv_classification", config=cfg)
        >>> logger.log({"train_loss": 0.5, "val_f1": 0.8}, step=10)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        WandB 로거 초기화

        Args:
            project: WandB 프로젝트 이름 (예: "cv_classification")
            name: 실험 이름 (예: "efficientnet_b0_v1")
            config: 하이퍼파라미터 설정 (Hydra config 등)
            tags: 실험 태그 리스트 (예: ["baseline", "augmentation"])
            notes: 실험 노트/메모
            enabled: WandB 활성화 여부 (False면 로깅 안 함)
        """
        self.enabled = enabled

        if self.enabled:
            # WandB 초기화
            wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
            )
            print(f"✅ WandB 초기화 완료: {project}/{name}")
        else:
            print("⚠️  WandB 비활성화됨 (enabled=False)")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        메트릭 로깅

        Args:
            metrics: 로깅할 메트릭 딕셔너리
                    예: {"train_loss": 0.5, "val_f1": 0.8, "lr": 0.001}
            step: 현재 step/epoch (옵션)

        예시:
            >>> logger.log({"train_loss": 0.5, "val_macro_f1": 0.82}, step=10)
        """
        if self.enabled:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[list] = None,
        step: Optional[int] = None,
    ):
        """
        Confusion Matrix 시각화 및 로깅

        Args:
            confusion_matrix: Confusion matrix 배열 (shape: [num_classes, num_classes])
            class_names: 클래스 이름 리스트 (예: ["target_0", "target_1", ...])
            step: 현재 step/epoch

        예시:
            >>> cm = get_confusion_matrix(preds, targets)
            >>> logger.log_confusion_matrix(cm, class_names=[f"class_{i}" for i in range(17)])
        """
        if not self.enabled:
            return

        # 클래스 이름이 없으면 기본값 생성
        if class_names is None:
            num_classes = confusion_matrix.shape[0]
            class_names = [f'target_{i}' for i in range(num_classes)]

        # Confusion matrix 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title('Confusion Matrix')
        plt.ylabel('실제 클래스')
        plt.xlabel('예측 클래스')

        # WandB에 업로드
        if step is not None:
            wandb.log({"confusion_matrix": wandb.Image(plt)}, step=step)
        else:
            wandb.log({"confusion_matrix": wandb.Image(plt)})

        plt.close()

    def log_class_metrics(self, class_f1_dict: Dict[str, float], step: Optional[int] = None):
        """
        클래스별 메트릭 로깅

        각 클래스별 F1 score를 개별적으로 로깅합니다.

        Args:
            class_f1_dict: 클래스별 F1 딕셔너리
                          예: {"class_0": 0.85, "class_1": 0.92, ...}
            step: 현재 step/epoch

        예시:
            >>> class_f1 = calculate_class_f1(preds, targets)
            >>> logger.log_class_metrics(class_f1, step=10)
        """
        if self.enabled:
            # "class_metrics/" 접두사를 붙여서 그룹화
            metrics = {f"class_metrics/{k}": v for k, v in class_f1_dict.items()}
            self.log(metrics, step=step)

    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        """
        모델 파라미터 추적

        모델의 gradient와 파라미터를 WandB에 기록합니다.
        (주의: 로깅 오버헤드가 있으므로 필요시에만 사용)

        Args:
            model: PyTorch 모델
            log: 로깅 타입 ("gradients", "parameters", "all")
            log_freq: 로깅 빈도 (매 N step마다)

        예시:
            >>> logger.watch_model(model, log="all", log_freq=100)
        """
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        하이퍼파라미터 추가 로깅

        초기화 후 추가 하이퍼파라미터를 로깅할 때 사용합니다.

        Args:
            params: 하이퍼파라미터 딕셔너리
        """
        if self.enabled:
            wandb.config.update(params)

    def save_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """
        파일을 WandB Artifact로 저장

        모델 체크포인트, 설정 파일 등을 버전 관리합니다.

        Args:
            file_path: 저장할 파일 경로
            name: Artifact 이름
            artifact_type: Artifact 타입 (예: "model", "dataset", "config")

        예시:
            >>> logger.save_artifact("best.pth", "best_model", "model")
        """
        if self.enabled:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            print(f"✅ Artifact 저장 완료: {name} ({file_path})")

    def finish(self):
        """
        WandB run 종료

        학습이 끝나면 반드시 호출하여 WandB run을 정상적으로 종료합니다.

        예시:
            >>> logger.finish()
        """
        if self.enabled:
            wandb.finish()
            print("✅ WandB run 종료")


def create_logger_from_config(cfg) -> WandBLogger:
    """
    Hydra config로부터 WandB 로거 생성

    Hydra config의 wandb 섹션을 읽어서 WandBLogger를 초기화합니다.

    Args:
        cfg: Hydra config 객체 (cfg.wandb 섹션 필요)

    Returns:
        초기화된 WandBLogger 인스턴스

    예시:
        >>> @hydra.main(config_path="configs", config_name="config")
        >>> def main(cfg):
        >>>     logger = create_logger_from_config(cfg)
    """
    # Config에서 WandB 설정 추출
    wandb_cfg = cfg.get('wandb', {})

    logger = WandBLogger(
        project=wandb_cfg.get('project', 'cv_classification'),
        name=wandb_cfg.get('name', None),  # 'experiment_name' → 'name'으로 수정
        config=dict(cfg),  # Hydra config 전체를 로깅
        tags=wandb_cfg.get('tags', None),
        notes=wandb_cfg.get('notes', None),
        enabled=wandb_cfg.get('enabled', True),
    )

    return logger
