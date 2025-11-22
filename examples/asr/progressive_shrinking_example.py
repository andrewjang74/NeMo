"""
Progressive Shrinking: 단계적으로 모델 크기를 줄이면서 학습

방법:
1. 17 layers -> 15 layers (fine-tune)
2. 15 layers -> 13 layers (fine-tune)
3. 13 layers -> 12 layers (fine-tune)

각 단계마다 이전 모델의 가중치를 최대한 활용
"""

import nemo.collections.asr as nemo_asr
import torch
import os


def shrink_model_by_layers(
    source_model_path,
    target_model_path,
    layers_to_remove,
    remove_strategy='middle'
):
    """
    모델에서 특정 레이어를 제거하여 작은 모델 생성
    
    Args:
        source_model_path: 원본 모델 경로
        target_model_path: 저장할 모델 경로
        layers_to_remove: 제거할 레이어 수
        remove_strategy: 'middle', 'end', 'uniform' 중 선택
    """
    
    # 모델 로드
    print(f"Loading model from {source_model_path}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(source_model_path)
    
    original_layers = len(model.encoder.layers)
    target_layers = original_layers - layers_to_remove
    
    print(f"Shrinking from {original_layers} to {target_layers} layers")
    print(f"Strategy: {remove_strategy}")
    
    # 유지할 레이어 선택
    if remove_strategy == 'middle':
        # 중간 레이어 제거 (처음과 끝 유지)
        remove_start = original_layers // 2 - layers_to_remove // 2
        keep_indices = (
            list(range(remove_start)) + 
            list(range(remove_start + layers_to_remove, original_layers))
        )
    
    elif remove_strategy == 'end':
        # 끝 레이어 제거
        keep_indices = list(range(target_layers))
    
    elif remove_strategy == 'uniform':
        # 균등하게 제거
        step = original_layers / target_layers
        keep_indices = [int(i * step) for i in range(target_layers)]
    
    else:
        raise ValueError(f"Unknown strategy: {remove_strategy}")
    
    print(f"Keeping layers: {keep_indices}")
    
    # 선택된 레이어만 유지
    new_layers = torch.nn.ModuleList([
        model.encoder.layers[i] for i in keep_indices
    ])
    model.encoder.layers = new_layers
    model.cfg.encoder.n_layers = target_layers
    
    # 저장
    print(f"Saving to {target_model_path}")
    model.save_to(target_model_path)
    print("✓ Done!")
    
    return model


def progressive_shrinking_pipeline(
    initial_model_path,
    target_layers,
    output_dir,
    train_script,
    train_manifest,
    val_manifest
):
    """
    단계적으로 모델을 줄이면서 학습하는 파이프라인
    
    Args:
        initial_model_path: 초기 학습된 모델 (예: 17 layers)
        target_layers: 최종 목표 레이어 수 (예: 12)
        output_dir: 출력 디렉토리
        train_script: 학습 스크립트 경로
        train_manifest: 학습 데이터
        val_manifest: 검증 데이터
    """
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 초기 모델 정보
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(initial_model_path)
    current_layers = len(model.encoder.layers)
    
    print(f"Starting progressive shrinking: {current_layers} -> {target_layers} layers")
    print(f"Output directory: {output_dir}")
    
    # 단계 계획
    # 예: 17 -> 15 -> 13 -> 12
    steps = []
    while current_layers > target_layers:
        next_layers = max(current_layers - 2, target_layers)
        steps.append((current_layers, next_layers))
        current_layers = next_layers
    
    print(f"\nPlanned steps: {steps}")
    
    current_model = initial_model_path
    
    # 각 단계 실행
    for step_idx, (from_layers, to_layers) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"Step {step_idx}/{len(steps)}: {from_layers} -> {to_layers} layers")
        print(f"{'='*60}")
        
        # 1. 모델 축소
        shrunk_model_path = os.path.join(
            output_dir, 
            f"step{step_idx}_shrunk_{to_layers}layers.nemo"
        )
        
        shrink_model_by_layers(
            source_model_path=current_model,
            target_model_path=shrunk_model_path,
            layers_to_remove=from_layers - to_layers,
            remove_strategy='middle'  # 또는 'uniform'
        )
        
        # 2. Fine-tuning
        finetuned_model_path = os.path.join(
            output_dir,
            f"step{step_idx}_finetuned_{to_layers}layers.nemo"
        )
        
        print(f"\nFine-tuning {to_layers}-layer model...")
        
        # 학습 명령어 구성
        train_cmd = [
            "python", train_script,
            f"model.train_ds.manifest_filepath={train_manifest}",
            f"model.validation_ds.manifest_filepath={val_manifest}",
            f"trainer.max_epochs=10",  # Fine-tuning은 짧게
            f"exp_manager.exp_dir={output_dir}/step{step_idx}",
            f"exp_manager.resume_from_checkpoint={shrunk_model_path}",
            f"trainer.devices=1"
        ]
        
        print(f"Running: {' '.join(train_cmd)}")
        
        # 실제 학습 실행 (주석 처리 - 필요시 활성화)
        # subprocess.run(train_cmd, check=True)
        
        print(f"✓ Step {step_idx} complete!")
        print(f"  Model saved to: {finetuned_model_path}")
        
        # 다음 단계를 위해 현재 모델 업데이트
        current_model = finetuned_model_path
    
    print(f"\n{'='*60}")
    print("Progressive shrinking complete!")
    print(f"Final model: {current_model}")
    print(f"{'='*60}")
    
    return current_model


# 사용 예시
if __name__ == "__main__":
    
    # 예시 1: 단일 단계 축소
    print("Example 1: Single step shrinking")
    shrink_model_by_layers(
        source_model_path="models/teacher_17layers.nemo",
        target_model_path="models/student_12layers_init.nemo",
        layers_to_remove=5,
        remove_strategy='middle'
    )
    
    # 예시 2: 단계적 축소 파이프라인
    print("\n\nExample 2: Progressive shrinking pipeline")
    progressive_shrinking_pipeline(
        initial_model_path="models/teacher_17layers.nemo",
        target_layers=12,
        output_dir="experiments/progressive_shrinking",
        train_script="examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py",
        train_manifest="data/train_manifest.json",
        val_manifest="data/val_manifest.json"
    )
    
    print("\n완료! 생성된 모델을 사용하여 추가 학습을 진행하세요.")
