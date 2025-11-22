#!/usr/bin/env python3
"""
Quick Start Script for Knowledge Distillation

이 스크립트는 Knowledge Distillation을 빠르게 시작할 수 있도록 도와줍니다.

사용 방법:
    python quick_start_distillation.py \
        --teacher_model teacher_17layers.nemo \
        --train_manifest train.json \
        --val_manifest val.json \
        --tokenizer_dir tokenizer \
        --output_dir experiments/my_distillation

선택적 파라미터:
    --alpha 0.6              # Student loss 가중치 (기본: 0.5)
    --temperature 2.0        # Distillation temperature (기본: 2.0)
    --max_epochs 100         # 최대 epoch (기본: 100)
    --devices 1              # GPU 개수 (기본: 1)
    --batch_size 16          # Batch size (기본: 16)
"""

import argparse
import os
import subprocess
import sys


def check_file_exists(filepath, description):
    """파일 존재 확인"""
    if not os.path.exists(filepath):
        print(f"❌ Error: {description} not found: {filepath}")
        return False
    print(f"✓ {description} found: {filepath}")
    return True


def check_directory_exists(dirpath, description):
    """디렉토리 존재 확인"""
    if not os.path.isdir(dirpath):
        print(f"❌ Error: {description} not found: {dirpath}")
        return False
    print(f"✓ {description} found: {dirpath}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Quick Start for Knowledge Distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python quick_start_distillation.py \\
        --teacher_model models/teacher_17layers.nemo \\
        --train_manifest data/train.json \\
        --val_manifest data/val.json \\
        --tokenizer_dir tokenizer \\
        --output_dir experiments/distillation_run1
        
    python quick_start_distillation.py \\
        --teacher_model models/teacher.nemo \\
        --train_manifest data/train.json \\
        --val_manifest data/val.json \\
        --tokenizer_dir tokenizer \\
        --output_dir experiments/exp1 \\
        --alpha 0.6 \\
        --temperature 2.5 \\
        --max_epochs 150 \\
        --devices 2
        """
    )
    
    # 필수 파라미터
    parser.add_argument(
        '--teacher_model',
        type=str,
        required=True,
        help='Teacher 모델 경로 (.nemo 파일)'
    )
    parser.add_argument(
        '--train_manifest',
        type=str,
        required=True,
        help='학습 데이터 manifest 파일 경로'
    )
    parser.add_argument(
        '--val_manifest',
        type=str,
        required=True,
        help='검증 데이터 manifest 파일 경로'
    )
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        required=True,
        help='Tokenizer 디렉토리 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='출력 디렉토리 (실험 결과 저장)'
    )
    
    # 선택적 파라미터
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Student loss 가중치 (0.0~1.0, 기본: 0.5)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='Distillation temperature (기본: 2.0)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='최대 epoch 수 (기본: 100)'
    )
    parser.add_argument(
        '--devices',
        type=int,
        default=1,
        help='사용할 GPU 개수 (기본: 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (기본: 16)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3.0,
        help='Learning rate (기본: 3.0)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=5000,
        help='Warmup steps (기본: 5000)'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Weights & Biases 로깅 사용'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='asr_distillation',
        help='W&B 프로젝트 이름 (기본: asr_distillation)'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default=None,
        help='W&B 실험 이름 (기본: output_dir 이름)'
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=32,
        choices=[16, 32],
        help='학습 precision (16 or 32, 기본: 32)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='실제 실행 없이 명령어만 출력'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Knowledge Distillation Quick Start")
    print("=" * 70)
    print()
    
    # 1. 파일/디렉토리 존재 확인
    print("📋 Step 1: 파일 확인")
    print("-" * 70)
    
    all_ok = True
    all_ok &= check_file_exists(args.teacher_model, "Teacher model")
    all_ok &= check_file_exists(args.train_manifest, "Train manifest")
    all_ok &= check_file_exists(args.val_manifest, "Validation manifest")
    all_ok &= check_directory_exists(args.tokenizer_dir, "Tokenizer directory")
    
    if not all_ok:
        print("\n❌ 일부 파일/디렉토리를 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    
    print()
    
    # 2. 설정 확인
    print("⚙️  Step 2: 설정 확인")
    print("-" * 70)
    print(f"Teacher Model:      {args.teacher_model}")
    print(f"Train Manifest:     {args.train_manifest}")
    print(f"Val Manifest:       {args.val_manifest}")
    print(f"Tokenizer:          {args.tokenizer_dir}")
    print(f"Output Directory:   {args.output_dir}")
    print()
    print(f"Alpha:              {args.alpha}")
    print(f"Temperature:        {args.temperature}")
    print(f"Max Epochs:         {args.max_epochs}")
    print(f"Devices:            {args.devices}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Learning Rate:      {args.learning_rate}")
    print(f"Warmup Steps:       {args.warmup_steps}")
    print(f"Precision:          {args.precision}")
    print(f"Use W&B:            {args.use_wandb}")
    print()
    
    # 3. 명령어 생성
    print("🚀 Step 3: 학습 명령어 생성")
    print("-" * 70)
    
    # 스크립트 경로 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(
        script_dir,
        "asr_hybrid_transducer_ctc",
        "speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py"
    )
    
    if not os.path.exists(training_script):
        print(f"❌ Training script not found: {training_script}")
        sys.exit(1)
    
    # 명령어 구성
    cmd = [
        "python",
        training_script,
        "--config-path=../conf/fastconformer/hybrid_transducer_ctc",
        "--config-name=fastconformer_hybrid_transducer_ctc_bpe_small",
        f"model.train_ds.manifest_filepath={args.train_manifest}",
        f"model.validation_ds.manifest_filepath={args.val_manifest}",
        f"model.tokenizer.dir={args.tokenizer_dir}",
        "model.tokenizer.type=bpe",
        f"teacher_model_path={args.teacher_model}",
        f"distillation.alpha={args.alpha}",
        f"distillation.temperature={args.temperature}",
        "distillation.use_encoder_distill=true",
        "distillation.use_logits_distill=true",
        f"trainer.devices={args.devices}",
        f"trainer.max_epochs={args.max_epochs}",
        f"trainer.precision={args.precision}",
        f"model.train_ds.batch_size={args.batch_size}",
        f"model.validation_ds.batch_size={args.batch_size}",
        f"model.optim.lr={args.learning_rate}",
        f"model.optim.sched.warmup_steps={args.warmup_steps}",
        f"exp_manager.exp_dir={args.output_dir}",
    ]
    
    # W&B 설정
    if args.use_wandb:
        wandb_name = args.wandb_name or os.path.basename(args.output_dir)
        cmd.extend([
            "exp_manager.create_wandb_logger=true",
            f"exp_manager.wandb_logger_kwargs.project={args.wandb_project}",
            f"exp_manager.wandb_logger_kwargs.name={wandb_name}",
        ])
    
    # 명령어 출력
    print("\n실행할 명령어:")
    print("-" * 70)
    print(" \\\n  ".join(cmd))
    print("-" * 70)
    print()
    
    # 4. 실행
    if args.dry_run:
        print("🔍 Dry run mode: 명령어만 출력하고 종료합니다.")
        print("\n실제 실행하려면 --dry_run 옵션을 제거하세요:")
        print(f"  python {' '.join(sys.argv).replace(' --dry_run', '')}")
        return
    
    print("✓ 모든 확인 완료!")
    print()
    
    # 사용자 확인
    response = input("학습을 시작하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    print()
    print("=" * 70)
    print("🚀 학습 시작!")
    print("=" * 70)
    print()
    
    # 실행
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 70)
        print("✓ 학습 완료!")
        print("=" * 70)
        print(f"\n결과 확인:")
        print(f"  TensorBoard: tensorboard --logdir {args.output_dir}")
        print(f"  로그 파일: {args.output_dir}/nemo_log_*.txt")
        print(f"  체크포인트: {args.output_dir}/checkpoints/")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print("❌ 학습 중 오류 발생!")
        print("=" * 70)
        print(f"\n오류 코드: {e.returncode}")
        print("\n로그를 확인하세요:")
        print(f"  tail -f {args.output_dir}/nemo_log_*.txt")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⚠️  사용자에 의해 중단되었습니다.")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
