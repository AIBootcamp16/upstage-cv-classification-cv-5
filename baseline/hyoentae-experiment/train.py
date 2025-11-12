"""
Train Script

Hydra와 WandB를 사용한 모델 학습 스크립트

사용법:
    # 단일 모델 학습
    python train.py model=efficientnet_b0

    # 강력한 증강 사용
    python train.py model=efficientnet_b0 augmentation=strong

    # 4개 모델 동시 실험 (Hydra multi-run)
    python train.py -m model=efficientnet_b0,efficientnet_b3,resnet50,convnext_tiny

    # Augmentation 비교 실험
    python train.py -m augmentation=default,strong
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import os
import random
import numpy as np
from pathlib import Path

from src.models.classifier import create_model_from_config
from src.models.losses import create_loss_from_config
from src.data.dataset import create_train_val_datasets, create_dataloaders, ClassConditionalAugraphyDataset
from src.data.transforms import create_transforms_from_config, create_augraphy_pipeline
from src.trainer import Trainer, create_optimizer, create_scheduler
from src.utils.logger import create_logger_from_config
from src.utils.checkpoint import CheckpointManager
from src.utils.mixup import create_mixup_cutmix_from_config
from src.utils.kfold import create_kfold_splits, print_kfold_info, get_fold_save_dir, save_kfold_summary


def set_seed(seed: int = 42):
    """
    재현성을 위한 랜덤 시드 고정

    Args:
        seed: 랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN 결정론적 동작 (속도 약간 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 랜덤 시드 고정: {seed}")


def get_device(cfg) -> str:
    """
    사용 가능한 디바이스 확인

    CUDA 사용 가능하면 GPU, 아니면 CPU 사용

    Args:
        cfg: Hydra config

    Returns:
        디바이스 문자열 ('cuda' 또는 'cpu')
    """
    device_from_cfg = cfg.get('device', 'cuda')

    if device_from_cfg == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        if device_from_cfg == 'cuda':
            print("⚠️  CUDA를 사용할 수 없습니다. CPU로 fallback합니다.")
        print("✅ CPU 사용")

    return device


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    메인 학습 함수

    Hydra가 자동으로 config를 로드하고 현재 작업 디렉토리를
    outputs/YYYY-MM-DD/HH-MM-SS/ 로 변경합니다.

    Args:
        cfg: Hydra config
    """
    # Config 출력
    print("\n" + "="*60)
    print("⚙️  Config")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")

    # 랜덤 시드 고정
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # 디바이스 설정
    device = get_device(cfg)

    # 데이터 경로
    data_dir = Path(cfg.data.data_dir)
    train_csv = data_dir / cfg.data.train_csv
    train_img_dir = data_dir / cfg.data.train_dir

    print(f"\n📂 데이터 경로:")
    print(f"   CSV: {train_csv}")
    print(f"   Images: {train_img_dir}\n")

    # K-Fold 사용 여부 확인
    use_kfold = cfg.train.get('k_fold', {}).get('enabled', False)

    if use_kfold:
        # ========== K-Fold Cross Validation 모드 ==========
        print(f"\n🔀 K-Fold Cross Validation 모드 활성화")

        n_splits = cfg.train.k_fold.get('n_splits', 5)
        shuffle_kfold = cfg.train.k_fold.get('shuffle', True)

        # K-Fold 분할 생성
        kfold_splits = create_kfold_splits(
            train_csv=str(train_csv),
            n_splits=n_splits,
            shuffle=shuffle_kfold,
            random_state=seed
        )

        # K-Fold 정보 출력
        print_kfold_info(kfold_splits, str(train_csv))

        # Hydra의 실제 출력 디렉토리 가져오기
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir

        # 각 Fold별로 학습
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n{'='*60}")
            print(f"🔀 Fold {fold_idx+1}/{n_splits} 학습 시작")
            print(f"{'='*60}")

            # Fold별 저장 디렉토리
            fold_save_dir = get_fold_save_dir(hydra_output_dir, fold_idx)
            fold_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Fold {fold_idx} 저장 디렉토리: {fold_save_dir}")

            # Transforms 생성
            train_transform = create_transforms_from_config(cfg, mode='train')
            val_transform = create_transforms_from_config(cfg, mode='valid')

            # Dataset 생성 (임시 CSV 생성)
            from src.data.dataset import DocumentDataset, ClassConditionalAugraphyDataset
            from src.data.transforms import create_augraphy_pipeline
            import pandas as pd
            import tempfile

            df = pd.read_csv(str(train_csv))

            # 임시 CSV 파일 생성 (Fold별)
            train_fold_df = df.iloc[train_idx]
            val_fold_df = df.iloc[val_idx]

            # 임시 디렉토리에 저장
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_train.csv') as train_csv_file:
                train_fold_df.to_csv(train_csv_file.name, index=False)
                train_csv_path = train_csv_file.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_val.csv') as val_csv_file:
                val_fold_df.to_csv(val_csv_file.name, index=False)
                val_csv_path = val_csv_file.name

            # Augraphy 설정 확인
            use_conditional_augraphy = False
            augraphy_pipeline = None
            target_classes = None
            augraphy_probability = 0.7

            augraphy_cfg = cfg.get('augmentation', {}).get('augraphy', {})
            if augraphy_cfg.get('enabled', False):
                try:
                    import augraphy
                    AUGRAPHY_AVAILABLE = True
                except ImportError:
                    AUGRAPHY_AVAILABLE = False

                if AUGRAPHY_AVAILABLE:
                    target_classes = augraphy_cfg.get('target_classes', None)
                    if target_classes is not None:
                        # target_classes가 설정되어 있으면 ClassConditionalAugraphyDataset 사용
                        use_conditional_augraphy = True
                        strength = augraphy_cfg.get('strength', 'medium')
                        augraphy_probability = augraphy_cfg.get('probability', 0.7)
                        augraphy_pipeline = create_augraphy_pipeline(strength=strength)
                        print(f"   🎯 ClassConditionalAugraphyDataset 사용 (target_classes={target_classes})")

            # Dataset 생성
            if use_conditional_augraphy:
                # ClassConditionalAugraphyDataset 사용
                train_dataset_fold = ClassConditionalAugraphyDataset(
                    csv_path=train_csv_path,
                    img_dir=str(train_img_dir),
                    base_transform=train_transform,
                    augraphy_pipeline=augraphy_pipeline,
                    target_classes=target_classes,
                    augraphy_probability=augraphy_probability,
                    is_test=False
                )
            else:
                # 일반 DocumentDataset 사용
                train_dataset_fold = DocumentDataset(
                    csv_path=train_csv_path,
                    img_dir=str(train_img_dir),
                    transform=train_transform
                )

            val_dataset_fold = DocumentDataset(
                csv_path=val_csv_path,
                img_dir=str(train_img_dir),
                transform=val_transform
            )

            # DataLoader 생성
            train_loader, val_loader = create_dataloaders(
                train_dataset=train_dataset_fold,
                val_dataset=val_dataset_fold,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.data.get('num_workers', 4),
            )

            # 모델 생성 (각 fold마다 새로 생성)
            model = create_model_from_config(cfg, num_classes=cfg.data.num_classes)

            # Loss 함수 (Focal Loss, Label Smoothing 등)
            # label_smoothing을 cfg.train에서 cfg.loss로 전달
            if not hasattr(cfg, 'loss'):
                cfg.loss = {}
            if 'label_smoothing' not in cfg.loss and 'label_smoothing' in cfg.train:
                OmegaConf.set_struct(cfg, False)
                cfg.loss.label_smoothing = cfg.train.label_smoothing
                OmegaConf.set_struct(cfg, True)
            criterion = create_loss_from_config(cfg, device=device)

            # Optimizer
            optimizer = create_optimizer(model, cfg)

            # Scheduler
            scheduler = create_scheduler(optimizer, cfg, num_epochs=cfg.train.epochs)

            # WandB Logger (Fold별)
            model_name = cfg.model.name
            aug_type = cfg.get('augmentation', {}).get('name', 'default')

            # K-Fold용 실험 이름 (fold 번호 추가)
            if cfg.wandb.get('name') is None:
                fold_experiment_name = f"{model_name}_{aug_type}_fold{fold_idx}"
            else:
                fold_experiment_name = f"{cfg.wandb.name}_fold{fold_idx}"

            # Config 복사해서 이름만 변경
            fold_cfg = OmegaConf.to_container(cfg, resolve=True)
            fold_cfg = OmegaConf.create(fold_cfg)
            OmegaConf.set_struct(fold_cfg, False)
            fold_cfg.wandb.name = fold_experiment_name
            fold_cfg.wandb.tags = [model_name, aug_type, f"fold_{fold_idx}"]
            OmegaConf.set_struct(fold_cfg, True)

            logger = create_logger_from_config(fold_cfg)

            # Checkpoint Manager (Fold별)
            use_gen_score = cfg.train.get('use_generalization_score', True)
            overfitting_penalty = cfg.train.get('overfitting_penalty', 0.3)

            checkpoint_manager = CheckpointManager(
                save_dir=str(fold_save_dir),
                metric_name="macro_f1",
                mode="max",
                patience=cfg.train.early_stopping.patience,
                verbose=True,
                use_generalization_score=use_gen_score,
                overfitting_penalty=overfitting_penalty,
            )

            # MixUp/CutMix
            mixup_cutmix = create_mixup_cutmix_from_config(cfg)
            if fold_idx == 0 and mixup_cutmix is not None:
                # Fold 0에서만 출력 (중복 방지)
                mixup_alpha = cfg.train.get('mixup_alpha', 0.0)
                cutmix_alpha = cfg.train.get('cutmix_alpha', 0.0)
                print(f"✅ MixUp/CutMix 활성화 (mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha})")

            # Mixed Precision Training
            use_amp = cfg.train.get('use_amp', False)

            # Trainer 생성
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                mixup_cutmix=mixup_cutmix,
                use_amp=use_amp,
                model_config=dict(cfg.model),
            )

            # 학습 시작
            trainer.train(num_epochs=cfg.train.epochs)

            # WandB 종료
            if logger is not None:
                logger.finish()

            # Fold 결과 저장
            fold_result = {
                'fold': fold_idx,
                'best_epoch': checkpoint_manager.get_best_epoch(),
                'val_f1': checkpoint_manager.get_best_metric(),
                'train_f1': checkpoint_manager.get_best_train_metric(),
            }
            fold_results.append(fold_result)

            print(f"\n✅ Fold {fold_idx} 학습 완료!")
            print(f"   Best Epoch: {fold_result['best_epoch']}")
            print(f"   Val Macro F1: {fold_result['val_f1']:.4f}")
            if fold_result['train_f1'] is not None:
                print(f"   Train Macro F1: {fold_result['train_f1']:.4f}")

        # K-Fold 전체 결과 요약
        save_kfold_summary(hydra_output_dir, fold_results)

        return  # K-Fold 모드는 여기서 종료

    # ========== 기존 단일 모델 학습 모드 (K-Fold 비활성화) ==========
    # Transforms 생성
    train_transform = create_transforms_from_config(cfg, mode='train')
    val_transform = create_transforms_from_config(cfg, mode='valid')

    # Dataset 생성 (cfg 전달하여 Augraphy target_classes 지원)
    train_dataset, val_dataset = create_train_val_datasets(
        csv_path=str(train_csv),
        img_dir=str(train_img_dir),
        train_transform=train_transform,
        val_transform=val_transform,
        val_split=cfg.data.val_split,
        random_state=seed,
        cfg=cfg,  # Augraphy target_classes 지원
    )

    # DataLoader 생성
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.get('num_workers', 4),
    )

    # 모델 생성
    model = create_model_from_config(cfg, num_classes=cfg.data.num_classes)

    # Loss 함수 (Focal Loss, Label Smoothing 등)
    # label_smoothing을 cfg.train에서 cfg.loss로 전달
    if not hasattr(cfg, 'loss'):
        cfg.loss = {}
    if 'label_smoothing' not in cfg.loss and 'label_smoothing' in cfg.train:
        OmegaConf.set_struct(cfg, False)
        cfg.loss.label_smoothing = cfg.train.label_smoothing
        OmegaConf.set_struct(cfg, True)
    criterion = create_loss_from_config(cfg, device=device)

    # Optimizer
    optimizer = create_optimizer(model, cfg)

    # Scheduler
    scheduler = create_scheduler(optimizer, cfg, num_epochs=cfg.train.epochs)

    # WandB Logger
    # 실험 이름 자동 생성: 모델명_증강타입_v버전 (예: efficientnet_b0_default_v1)
    model_name = cfg.model.name
    aug_type = cfg.get('augmentation', {}).get('name', 'default')

    # Config의 wandb.name이 None이면 자동 생성
    if cfg.wandb.get('name') is None:
        # WandB API로 기존 run 개수 확인해서 버전 넘버링
        import wandb as wandb_module

        # 임시로 API만 초기화 (실제 run은 생성 안 함)
        api = wandb_module.Api()
        entity = cfg.wandb.get('entity', None)
        if entity is None:
            entity = api.default_entity
        project_path = f"{entity}/{cfg.wandb.project}"

        try:
            # 같은 이름 패턴의 run 개수 세기
            base_name = f"{model_name}_{aug_type}"
            runs = api.runs(project_path)

            # 같은 base_name으로 시작하는 run 개수 세기
            version_count = 0
            for run in runs:
                if run.name and run.name.startswith(base_name):
                    version_count += 1

            # 버전 번호 (기존 개수 + 1)
            version = version_count + 1
            auto_experiment_name = f"{base_name}_v{version}"

        except Exception as e:
            # WandB API 접근 실패 시 (프로젝트 없음 등) 기본값 v1 사용
            print(f"⚠️  WandB API 접근 실패 (프로젝트 없거나 처음 실행): {e}")
            auto_experiment_name = f"{model_name}_{aug_type}_v1"

        OmegaConf.set_struct(cfg, False)  # 구조 수정 허용
        cfg.wandb.name = auto_experiment_name
        # 태그도 자동 업데이트
        cfg.wandb.tags = [model_name, aug_type]
        OmegaConf.set_struct(cfg, True)
        print(f"✅ WandB 실험 이름 자동 설정: {auto_experiment_name}")
        print(f"   태그: {[model_name, aug_type]}")
    else:
        # 사용자가 직접 지정한 이름 사용
        print(f"✅ WandB 실험 이름 (사용자 지정): {cfg.wandb.name}")
        # 태그가 비어있으면 자동 설정
        if not cfg.wandb.tags or len(cfg.wandb.tags) == 0:
            OmegaConf.set_struct(cfg, False)
            cfg.wandb.tags = [model_name, aug_type]
            OmegaConf.set_struct(cfg, True)
            print(f"   태그 자동 설정: {[model_name, aug_type]}")

    logger = create_logger_from_config(cfg)

    # Checkpoint Manager
    # Hydra의 실제 출력 디렉토리 가져오기
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"💾 체크포인트 저장 디렉토리: {hydra_output_dir}")

    # Generalization Score 설정 (config에서 읽기, 없으면 기본값)
    use_gen_score = cfg.train.get('use_generalization_score', True)
    overfitting_penalty = cfg.train.get('overfitting_penalty', 0.3)

    checkpoint_manager = CheckpointManager(
        save_dir=hydra_output_dir,
        metric_name="macro_f1",
        mode="max",
        patience=cfg.train.early_stopping.patience,
        verbose=True,
        use_generalization_score=use_gen_score,
        overfitting_penalty=overfitting_penalty,
    )

    if use_gen_score:
        print(f"🔍 Generalization Score 활성화 (과적합 페널티: {overfitting_penalty})")
    else:
        print(f"📊 기본 모드: Val Macro F1만 사용")

    # MixUp/CutMix (있으면)
    mixup_cutmix = create_mixup_cutmix_from_config(cfg)
    if mixup_cutmix is not None:
        mixup_alpha = cfg.train.get('mixup_alpha', 0.0)
        cutmix_alpha = cfg.train.get('cutmix_alpha', 0.0)
        print(f"✅ MixUp/CutMix 활성화 (mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha})")

    # Mixed Precision Training
    use_amp = cfg.train.get('use_amp', False)

    # Trainer 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        mixup_cutmix=mixup_cutmix,
        use_amp=use_amp,
        model_config=dict(cfg.model),  # 모델 config 전달 (추론 시 재현용)
    )

    # 학습 시작
    trainer.train(num_epochs=cfg.train.epochs)

    # WandB 종료
    if logger is not None:
        logger.finish()

    print("\n✅ 학습 완료!")
    print(f"   Best Epoch: {checkpoint_manager.get_best_epoch()}")

    # Best epoch의 Train & Val Macro F1 출력
    best_train_f1 = checkpoint_manager.get_best_train_metric()
    best_val_f1 = checkpoint_manager.get_best_metric()

    if best_train_f1 is not None:
        print(f"   Best Train Macro F1: {best_train_f1:.4f}")
        print(f"   Best Val Macro F1: {best_val_f1:.4f}")
        gap = best_train_f1 - best_val_f1
        print(f"   Train-Val Gap: {gap:+.4f} ({abs(gap)*100:.1f}%)")
    else:
        print(f"   Best Val Macro F1: {best_val_f1:.4f}")

    print(f"   Checkpoint: {hydra_output_dir}/best.pth")


if __name__ == "__main__":
    main()
