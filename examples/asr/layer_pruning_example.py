"""
Layer Pruning: 기존 학습된 모델에서 레이어를 제거하여 작은 모델 만들기

사용 방법:
1. 기존 학습된 모델 (.nemo 파일) 준비
2. 원하는 레이어 수 설정
3. 새로운 작은 모델로 저장
4. Fine-tuning 수행
"""

import nemo.collections.asr as nemo_asr
import torch
import copy

def prune_encoder_layers(teacher_model_path, student_model_path, keep_layers):
    """
    기존 모델에서 특정 레이어만 유지하여 작은 모델 생성
    
    Args:
        teacher_model_path: 기존 학습된 모델 경로 (.nemo)
        student_model_path: 저장할 작은 모델 경로 (.nemo)
        keep_layers: 유지할 레이어 인덱스 리스트 (예: [0,1,2,4,6,8,10,12,14,15,16])
    """
    
    # 1. 기존 모델 로드
    print(f"Loading teacher model from {teacher_model_path}")
    teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(teacher_model_path)
    
    # 2. 모델 구조 확인
    original_layers = len(teacher_model.encoder.layers)
    print(f"Original model has {original_layers} layers")
    print(f"Keeping {len(keep_layers)} layers: {keep_layers}")
    
    # 3. 선택된 레이어만 유지
    new_layers = torch.nn.ModuleList([
        teacher_model.encoder.layers[i] for i in keep_layers
    ])
    teacher_model.encoder.layers = new_layers
    
    # 4. 설정 업데이트
    teacher_model.cfg.encoder.n_layers = len(keep_layers)
    
    # 5. 작은 모델로 저장
    print(f"Saving pruned model to {student_model_path}")
    teacher_model.save_to(student_model_path)
    
    print(f"✓ Successfully created smaller model with {len(keep_layers)} layers")
    
    return teacher_model


def prune_uniformly(teacher_model_path, student_model_path, target_layers):
    """
    균등하게 레이어를 제거하여 작은 모델 생성
    
    Args:
        teacher_model_path: 기존 학습된 모델 경로
        student_model_path: 저장할 작은 모델 경로
        target_layers: 목표 레이어 수 (예: 12)
    """
    
    # 기존 모델 로드
    teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(teacher_model_path)
    original_layers = len(teacher_model.encoder.layers)
    
    # 균등하게 레이어 선택
    # 예: 17개 -> 12개로 줄이면, 5개를 균등하게 제거
    step = original_layers / target_layers
    keep_layers = [int(i * step) for i in range(target_layers)]
    
    print(f"Reducing from {original_layers} to {target_layers} layers")
    print(f"Keeping layers: {keep_layers}")
    
    return prune_encoder_layers(teacher_model_path, student_model_path, keep_layers)


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 17개 레이어 -> 12개 레이어로 균등하게 줄이기
    teacher_path = "path/to/your/trained_model.nemo"
    student_path = "path/to/save/pruned_model_12layers.nemo"
    
    # 방법 1: 균등하게 레이어 제거
    pruned_model = prune_uniformly(
        teacher_model_path=teacher_path,
        student_model_path=student_path,
        target_layers=12
    )
    
    # 방법 2: 특정 레이어만 선택 (첫 부분과 끝 부분 더 많이 유지)
    # keep_layers = [0, 1, 2, 3, 5, 7, 9, 11, 13, 14, 15, 16]
    # pruned_model = prune_encoder_layers(teacher_path, student_path, keep_layers)
    
    print("\n다음 단계:")
    print("1. 생성된 작은 모델로 fine-tuning 수행")
    print("2. 학습 스크립트에서 resume_from_checkpoint 사용")
