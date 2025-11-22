#!/usr/bin/env python3
"""
Progressive Shrinking - 자동화된 단계적 모델 축소 및 학습

이 스크립트는 큰 모델을 단계적으로 줄이면서 각 단계마다 fine-tuning을 수행합니다.

사용 방법:
    python progressive_shrinking_trainer.py \
        --teacher_model teacher_17layers.nemo \
        --target_layers 12 \
        --train_manifest train.json \
        --val_manifest val.json \
        --tokenizer_dir tokenizer \
        --output_dir experiments/progressive_shrinking

단계:
    17 layers → 15 layers (fine-tune 10 epochs)
    15 layers → 13 layers (fine-tune 10 epochs)
    13 layers → 12 layers (fine-tune 10 epochs)
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch


class ProgressiveShrinkingTrainer:
    """Progressive Shrinking을 자동화하는 클래스"""
    
    def __init__(
        self,
        teacher_model_path,
        target_layers,
        train_manifest,
        val_manifest,
        tokenizer_dir,
        output_dir,
        step_size=2,
        epochs_per_step=10,
        remove_strategy='uniform',
        learning_rate=1.0,
        devices=1,
        batch_size=16,
        precision=32
    ):
        self.teacher_model_path = teacher_model_path
        self.target_layers = target_layers
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.tokenizer_dir = tokenizer_dir
        self.output_dir = output_dir
        self.step_size = step_size
        self.epochs_per_step = epochs_per_step
        self.remove_strategy = remove_strategy
        self.learning_rate = learning_rate
        self.devices = devices
        self.batch_size = batch_size
        self.precision = precision
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 로그 파일
        self.log_file = os.path.join(output_dir, 'progressive_shrinking.log')
        
    def log(self, message):
        """로그 출력 및 파일 저장"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def get_model_info(self, model_path):
        """모델 정보 확인"""
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
        num_layers = len(model.encoder.layers)
        d_model = model.encoder.d_model
        return num_layers, d_model
    
    def shrink_model(self, source_path, target_path, target_layers):
        """
        모델 레이어 축소
        
        Args:
            source_path: 원본 모델 경로
            target_path: 저장할 모델 경로
            target_layers: 목표 레이어 수
        """
        self.log(f"\n{'='*70}")
        self.log(f"모델 축소: {source_path}")
        self.log(f"{'='*70}")
        
        # 모델 로드
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(source_path)
        original_layers = len(model.encoder.layers)
        
        self.log(f"원본 레이어 수: {original_layers}")
        self.log(f"목표 레이어 수: {target_layers}")
        self.log(f"제거 전략: {self.remove_strategy}")
        
        # 유지할 레이어 선택
        if self.remove_strategy == 'uniform':
            # 균등하게 선택
            step = original_layers / target_layers
            keep_indices = [int(i * step) for i in range(target_layers)]
        
        elif self.remove_strategy == 'middle':
            # 중간 레이어 제거
            layers_to_remove = original_layers - target_layers
            remove_start = original_layers // 2 - layers_to_remove // 2
            keep_indices = (
                list(range(remove_start)) + 
                list(range(remove_start + layers_to_remove, original_layers))
            )
        
        elif self.remove_strategy == 'end':
            # 끝 레이어 제거
            keep_indices = list(range(target_layers))
        
        else:
            raise ValueError(f"Unknown strategy: {self.remove_strategy}")
        
        self.log(f"유지할 레이어: {keep_indices}")
        
        # 레이어 축소
        new_layers = torch.nn.ModuleList([
            model.encoder.layers[i] for i in keep_indices
        ])
        model.encoder.layers = new_layers
        model.cfg.encoder.n_layers = target_layers
        
        # 저장
        self.log(f"저장 경로: {target_path}")
        model.save_to(target_path)
        self.log("✓ 모델 축소 완료!")
        
        return target_path
    
    def finetune_model(self, model_path, step_idx, num_layers, epochs):
        """
        모델 Fine-tuning
        
        Args:
            model_path: 학습할 모델 경로
            step_idx: 현재 단계 번호
            num_layers: 현재 레이어 수
            epochs: 학습 epoch 수
        """
        self.log(f"\n{'='*70}")
        self.log(f"Fine-tuning 시작: {num_layers} layers")
        self.log(f"{'='*70}")
        
        # 학습 스크립트 경로
        script_dir = Path(__file__).parent
        train_script = script_dir / "asr_hybrid_transducer_ctc" / "speech_to_text_hybrid_rnnt_ctc_bpe.py"
        
        if not train_script.exists():
            # 대체 경로 시도
            train_script = Path("examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py")
        
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")
        
        # 실험 디렉토리
        exp_dir = os.path.join(self.output_dir, f"step{step_idx}_finetune_{num_layers}layers")
        
        # 학습 명령어
        cmd = [
            "python", str(train_script),
            "--config-path=../conf/fastconformer/hybrid_transducer_ctc",
            "--config-name=fastconformer_hybrid_transducer_ctc_bpe",
            f"model.train_ds.manifest_filepath={self.train_manifest}",
            f"model.validation_ds.manifest_filepath={self.val_manifest}",
            f"model.tokenizer.dir={self.tokenizer_dir}",
            "model.tokenizer.type=bpe",
            f"+init_from_nemo_model={model_path}",
            f"trainer.devices={self.devices}",
            f"trainer.max_epochs={epochs}",
            f"trainer.precision={self.precision}",
            f"model.train_ds.batch_size={self.batch_size}",
            f"model.validation_ds.batch_size={self.batch_size}",
            f"model.optim.lr={self.learning_rate}",
            "model.optim.sched.warmup_steps=1000",  # Fine-tuning은 짧게
            f"exp_manager.exp_dir={exp_dir}",
            "exp_manager.create_checkpoint_callback=true",
            "exp_manager.checkpoint_callback_params.save_top_k=1",
            "exp_manager.checkpoint_callback_params.monitor=val_wer",
            "exp_manager.checkpoint_callback_params.mode=min",
        ]
        
        self.log(f"\n실행 명령어:")
        self.log(" \\\n  ".join(cmd))
        self.log("")
        
        # 학습 실행
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            self.log("✓ Fine-tuning 완료!")
            
            # Best checkpoint 찾기
            checkpoint_dir = os.path.join(exp_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.nemo')]
                if checkpoints:
                    # 가장 최근 또는 best checkpoint
                    best_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
                    self.log(f"Best checkpoint: {best_ckpt}")
                    return best_ckpt
            
            # Checkpoint를 못 찾으면 원본 모델 반환
            self.log("⚠️  Checkpoint를 찾을 수 없습니다. 원본 모델 사용.")
            return model_path
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Fine-tuning 실패: {e}")
            raise
    
    def plan_steps(self):
        """학습 단계 계획"""
        # 초기 모델 정보
        initial_layers, d_model = self.get_model_info(self.teacher_model_path)
        
        self.log(f"\n{'='*70}")
        self.log("Progressive Shrinking 계획")
        self.log(f"{'='*70}")
        self.log(f"초기 모델: {self.teacher_model_path}")
        self.log(f"초기 레이어: {initial_layers}")
        self.log(f"목표 레이어: {self.target_layers}")
        self.log(f"D-model: {d_model}")
        self.log(f"단계 크기: {self.step_size} layers")
        self.log(f"각 단계 epoch: {self.epochs_per_step}")
        
        # 단계 생성
        steps = []
        current = initial_layers
        while current > self.target_layers:
            next_layers = max(current - self.step_size, self.target_layers)
            steps.append((current, next_layers))
            current = next_layers
        
        self.log(f"\n계획된 단계:")
        for i, (from_l, to_l) in enumerate(steps, 1):
            self.log(f"  Step {i}: {from_l} → {to_l} layers")
        
        self.log(f"\n총 단계 수: {len(steps)}")
        self.log(f"예상 총 epoch: {len(steps) * self.epochs_per_step}")
        
        return steps
    
    def run(self):
        """전체 파이프라인 실행"""
        self.log("\n" + "="*70)
        self.log("Progressive Shrinking 시작")
        self.log("="*70)
        
        # 단계 계획
        steps = self.plan_steps()
        
        # 현재 모델 (시작은 teacher)
        current_model = self.teacher_model_path
        
        # 각 단계 실행
        for step_idx, (from_layers, to_layers) in enumerate(steps, 1):
            self.log(f"\n\n{'#'*70}")
            self.log(f"# Step {step_idx}/{len(steps)}: {from_layers} → {to_layers} layers")
            self.log(f"{'#'*70}")
            
            # 1. 모델 축소
            shrunk_model_path = os.path.join(
                self.output_dir,
                f"step{step_idx}_shrunk_{to_layers}layers.nemo"
            )
            
            self.shrink_model(
                source_path=current_model,
                target_path=shrunk_model_path,
                target_layers=to_layers
            )
            
            # 2. Fine-tuning
            finetuned_model = self.finetune_model(
                model_path=shrunk_model_path,
                step_idx=step_idx,
                num_layers=to_layers,
                epochs=self.epochs_per_step
            )
            
            # 다음 단계를 위해 업데이트
            current_model = finetuned_model
            
            self.log(f"\n✓ Step {step_idx} 완료!")
            self.log(f"  현재 모델: {current_model}")
        
        # 최종 결과
        self.log(f"\n\n{'='*70}")
        self.log("Progressive Shrinking 완료!")
        self.log(f"{'='*70}")
        self.log(f"최종 모델: {current_model}")
        self.log(f"로그 파일: {self.log_file}")
        
        # 결과 요약 저장
        summary = {
            'teacher_model': self.teacher_model_path,
            'final_model': current_model,
            'target_layers': self.target_layers,
            'steps': steps,
            'total_steps': len(steps),
            'epochs_per_step': self.epochs_per_step,
            'total_epochs': len(steps) * self.epochs_per_step,
        }
        
        summary_file = os.path.join(self.output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"요약 파일: {summary_file}")
        
        return current_model


def main():
    parser = argparse.ArgumentParser(
        description="Progressive Shrinking - 단계적 모델 축소 및 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python progressive_shrinking_trainer.py \\
        --teacher_model models/teacher_17layers.nemo \\
        --target_layers 12 \\
        --train_manifest data/train.json \\
        --val_manifest data/val.json \\
        --tokenizer_dir tokenizer \\
        --output_dir experiments/progressive_shrinking
        """
    )
    
    # 필수 파라미터
    parser.add_argument('--teacher_model', type=str, required=True,
                        help='Teacher 모델 경로 (.nemo)')
    parser.add_argument('--target_layers', type=int, required=True,
                        help='최종 목표 레이어 수')
    parser.add_argument('--train_manifest', type=str, required=True,
                        help='학습 데이터 manifest')
    parser.add_argument('--val_manifest', type=str, required=True,
                        help='검증 데이터 manifest')
    parser.add_argument('--tokenizer_dir', type=str, required=True,
                        help='Tokenizer 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='출력 디렉토리')
    
    # 선택적 파라미터
    parser.add_argument('--step_size', type=int, default=2,
                        help='각 단계마다 줄일 레이어 수 (기본: 2)')
    parser.add_argument('--epochs_per_step', type=int, default=10,
                        help='각 단계의 fine-tuning epoch (기본: 10)')
    parser.add_argument('--remove_strategy', type=str, default='uniform',
                        choices=['uniform', 'middle', 'end'],
                        help='레이어 제거 전략 (기본: uniform)')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='Learning rate (기본: 1.0)')
    parser.add_argument('--devices', type=int, default=1,
                        help='GPU 개수 (기본: 1)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (기본: 16)')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32],
                        help='학습 precision (기본: 32)')
    
    args = parser.parse_args()
    
    # 파일 확인
    if not os.path.exists(args.teacher_model):
        print(f"❌ Teacher model not found: {args.teacher_model}")
        sys.exit(1)
    
    if not os.path.exists(args.train_manifest):
        print(f"❌ Train manifest not found: {args.train_manifest}")
        sys.exit(1)
    
    if not os.path.exists(args.val_manifest):
        print(f"❌ Validation manifest not found: {args.val_manifest}")
        sys.exit(1)
    
    if not os.path.isdir(args.tokenizer_dir):
        print(f"❌ Tokenizer directory not found: {args.tokenizer_dir}")
        sys.exit(1)
    
    # Trainer 생성 및 실행
    trainer = ProgressiveShrinkingTrainer(
        teacher_model_path=args.teacher_model,
        target_layers=args.target_layers,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        step_size=args.step_size,
        epochs_per_step=args.epochs_per_step,
        remove_strategy=args.remove_strategy,
        learning_rate=args.learning_rate,
        devices=args.devices,
        batch_size=args.batch_size,
        precision=args.precision
    )
    
    try:
        final_model = trainer.run()
        print(f"\n✓ 성공! 최종 모델: {final_model}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
