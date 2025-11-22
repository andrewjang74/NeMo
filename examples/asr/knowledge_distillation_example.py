"""
Knowledge Distillation for ASR Models

Teacher 모델의 지식을 Student 모델에게 전달하는 학습 방법

주요 개념:
1. Teacher Loss: 큰 모델(Teacher)의 출력 분포를 작은 모델(Student)이 모방
2. Student Loss: 실제 정답(Ground Truth)과의 차이
3. Combined Loss: Teacher Loss + Student Loss
"""

import nemo.collections.asr as nemo_asr
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from nemo.core.classes import Loss
import copy


class DistillationLoss(Loss):
    """
    Knowledge Distillation Loss for ASR
    
    L_total = α * L_student + (1-α) * L_distill
    
    - L_student: 실제 정답과의 loss (CTC/RNNT)
    - L_distill: Teacher 출력과의 KL divergence
    - α: student loss의 가중치 (보통 0.5~0.7)
    """
    
    def __init__(self, teacher_model, alpha=0.5, temperature=2.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Teacher는 평가 모드로 고정
        self.alpha = alpha
        self.temperature = temperature
        
        # Teacher 파라미터 고정
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def forward(self, student_logits, teacher_logits, targets, student_loss):
        """
        Args:
            student_logits: Student 모델의 출력 logits
            teacher_logits: Teacher 모델의 출력 logits
            targets: 실제 정답
            student_loss: Student의 원래 loss (CTC/RNNT)
        """
        
        # Distillation Loss: KL Divergence between teacher and student
        # Temperature를 사용하여 soft targets 생성
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined Loss
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        
        return total_loss, {
            'student_loss': student_loss,
            'distill_loss': distill_loss,
            'total_loss': total_loss
        }


def create_student_model_from_config(config_path, teacher_model_path=None):
    """
    설정 파일로부터 Student 모델 생성
    
    Args:
        config_path: Student 모델 설정 파일 (작은 모델 설정)
        teacher_model_path: (Optional) Teacher 모델로부터 일부 가중치 초기화
    """
    from omegaconf import OmegaConf
    
    # Student 설정 로드
    cfg = OmegaConf.load(config_path)
    
    # Student 모델 생성
    student_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=cfg.model)
    
    # Teacher 모델이 있으면 일부 가중치 복사
    if teacher_model_path:
        print("Initializing student from teacher weights...")
        teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(teacher_model_path)
        
        # Encoder의 일부 레이어 복사 (예: 처음 몇 개 레이어)
        student_layers = len(student_model.encoder.layers)
        for i in range(student_layers):
            if i < len(teacher_model.encoder.layers):
                student_model.encoder.layers[i].load_state_dict(
                    teacher_model.encoder.layers[i].state_dict()
                )
        
        # Preprocessor 가중치 복사
        student_model.preprocessor.load_state_dict(teacher_model.preprocessor.state_dict())
        
        print(f"✓ Initialized {student_layers} layers from teacher model")
    
    return student_model


# 사용 예시: 학습 스크립트 수정
"""
기존 학습 스크립트를 다음과 같이 수정:

1. Teacher 모델 로드
2. Student 모델 생성 (작은 설정)
3. Distillation Loss 사용
4. 학습 수행

예시 코드:

# Teacher 모델 로드
teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "path/to/teacher_model.nemo"
)

# Student 모델 생성 (12 layers)
student_cfg = OmegaConf.load("fastconformer_hybrid_12layers.yaml")
student_model = create_student_model_from_config(
    config_path="fastconformer_hybrid_12layers.yaml",
    teacher_model_path="path/to/teacher_model.nemo"  # 초기화용
)

# Distillation Loss 설정
distill_loss = DistillationLoss(
    teacher_model=teacher_model,
    alpha=0.6,  # 60% student loss, 40% distillation loss
    temperature=2.0
)

# 학습 수행
trainer = Trainer(...)
trainer.fit(student_model)
"""


def train_with_distillation(
    teacher_model_path,
    student_config_path,
    train_manifest,
    val_manifest,
    output_dir,
    alpha=0.6,
    temperature=2.0
):
    """
    Knowledge Distillation을 사용한 학습 함수
    
    Args:
        teacher_model_path: Teacher 모델 경로 (.nemo)
        student_config_path: Student 모델 설정 파일
        train_manifest: 학습 데이터 manifest
        val_manifest: 검증 데이터 manifest
        output_dir: 출력 디렉토리
        alpha: Student loss 가중치
        temperature: Distillation temperature
    """
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    
    # 1. Teacher 모델 로드
    print("Loading teacher model...")
    teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(teacher_model_path)
    teacher_model.eval()
    
    # 2. Student 모델 생성
    print("Creating student model...")
    student_model = create_student_model_from_config(
        student_config_path,
        teacher_model_path=teacher_model_path
    )
    
    # 3. 데이터 설정
    cfg = OmegaConf.load(student_config_path)
    cfg.model.train_ds.manifest_filepath = train_manifest
    cfg.model.validation_ds.manifest_filepath = val_manifest
    
    # 4. Trainer 설정
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        default_root_dir=output_dir
    )
    
    # 5. 학습 시작
    print("Starting distillation training...")
    trainer.fit(student_model)
    
    # 6. 모델 저장
    student_model.save_to(f"{output_dir}/student_model_distilled.nemo")
    print(f"✓ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    # 사용 예시
    train_with_distillation(
        teacher_model_path="path/to/teacher_17layers.nemo",
        student_config_path="conf/fastconformer_hybrid_12layers.yaml",
        train_manifest="data/train_manifest.json",
        val_manifest="data/val_manifest.json",
        output_dir="experiments/distillation",
        alpha=0.6,
        temperature=2.0
    )
