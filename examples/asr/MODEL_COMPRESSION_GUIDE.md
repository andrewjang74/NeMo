# 기존 학습된 모델을 활용한 작은 모델 학습 가이드

## 📋 목차
1. [개요](#개요)
2. [방법 비교](#방법-비교)
3. [추천 워크플로우](#추천-워크플로우)
4. [단계별 실행 가이드](#단계별-실행-가이드)

---

## 개요

기존에 학습된 큰 모델(Teacher, 17 layers)을 활용하여 작은 모델(Student, 12 layers)을 효과적으로 학습하는 방법들을 제공합니다.

**목표**: 115M 파라미터 → 60M 파라미터 (약 50% 감소)

---

## 방법 비교

| 방법 | 난이도 | 성능 | 학습 시간 | 추천도 |
|------|--------|------|-----------|--------|
| **1. Layer Pruning** | ⭐ 쉬움 | 중간 | 짧음 | ⭐⭐⭐⭐ |
| **2. Knowledge Distillation** | ⭐⭐⭐ 어려움 | 높음 | 김 | ⭐⭐⭐⭐⭐ |
| **3. Progressive Shrinking** | ⭐⭐ 보통 | 높음 | 매우 김 | ⭐⭐⭐ |

---

## 추천 워크플로우

### 🥇 방법 1: Layer Pruning + Fine-tuning (가장 간단하고 실용적)

**장점:**
- 구현이 매우 간단
- 빠르게 결과 확인 가능
- Teacher 모델의 가중치를 직접 활용

**단점:**
- 성능이 다른 방법보다 약간 낮을 수 있음

**사용 시나리오:**
- 빠른 프로토타이핑이 필요한 경우
- 리소스가 제한적인 경우
- 첫 번째 시도로 적합

---

### 🥈 방법 2: Knowledge Distillation (최고 성능)

**장점:**
- 가장 높은 성능
- Teacher의 지식을 효과적으로 전달
- 학계에서 검증된 방법

**단점:**
- 구현이 복잡
- 학습 시간이 김
- 하이퍼파라미터 튜닝 필요

**사용 시나리오:**
- 최고 성능이 필요한 경우
- 충분한 학습 리소스가 있는 경우
- 프로덕션 배포용

---

### 🥉 방법 3: Progressive Shrinking (안정적)

**장점:**
- 안정적인 학습
- 각 단계별 성능 확인 가능

**단점:**
- 학습 시간이 매우 김
- 여러 단계 관리 필요

**사용 시나리오:**
- 안정성이 중요한 경우
- 단계별 분석이 필요한 경우

---

## 단계별 실행 가이드

### 방법 1: Layer Pruning + Fine-tuning

#### Step 1: 기존 모델에서 레이어 제거

```python
python examples/asr/layer_pruning_example.py
```

또는 직접 실행:

```python
import nemo.collections.asr as nemo_asr

# Teacher 모델 로드
teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "path/to/your/teacher_model_17layers.nemo"
)

# 12개 레이어만 유지 (균등하게 선택)
# 17개 중 12개 선택: [0, 1, 3, 5, 7, 9, 11, 13, 14, 15, 16]
keep_layers = [0, 1, 3, 5, 7, 9, 11, 13, 14, 15, 16]

new_layers = torch.nn.ModuleList([
    teacher_model.encoder.layers[i] for i in keep_layers
])
teacher_model.encoder.layers = new_layers
teacher_model.cfg.encoder.n_layers = 12

# 저장
teacher_model.save_to("models/student_12layers_init.nemo")
```

#### Step 2: Fine-tuning

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  model.train_ds.manifest_filepath=/path/to/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
  model.tokenizer.dir=/path/to/tokenizer \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  exp_manager.exp_dir=experiments/pruned_model_finetuning \
  +init_from_nemo_model=models/student_12layers_init.nemo
```

**Fine-tuning 팁:**
- Learning rate를 낮게 시작 (예: `model.optim.lr=1.0`)
- Warmup steps를 줄임 (예: `model.optim.sched.warmup_steps=1000`)
- Epoch 수는 원래 학습의 10-20% 정도

---

### 방법 2: Knowledge Distillation

#### Step 1: Student 모델 초기화

```python
from examples.asr.knowledge_distillation_example import create_student_model_from_config

student_model = create_student_model_from_config(
    config_path="examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe_small.yaml",
    teacher_model_path="path/to/teacher_model_17layers.nemo"
)

student_model.save_to("models/student_12layers_initialized.nemo")
```

#### Step 2: Distillation 학습

**참고**: 현재 NeMo ASR에는 내장 Distillation 기능이 없으므로, 
학습 스크립트를 수정하여 구현해야 합니다.

```python
# 학습 루프에서 다음과 같이 수정:

# Teacher 모델 로드 (평가 모드)
teacher_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "path/to/teacher_model.nemo"
)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 학습 중:
# 1. Teacher와 Student 모두에서 출력 얻기
# 2. Distillation loss 계산
# 3. Combined loss로 학습
```

---

### 방법 3: Progressive Shrinking

#### 전체 파이프라인 실행

```python
from examples.asr.progressive_shrinking_example import progressive_shrinking_pipeline

progressive_shrinking_pipeline(
    initial_model_path="models/teacher_17layers.nemo",
    target_layers=12,
    output_dir="experiments/progressive_shrinking",
    train_script="examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py",
    train_manifest="data/train_manifest.json",
    val_manifest="data/val_manifest.json"
)
```

**단계:**
1. 17 layers → 15 layers (fine-tune 10 epochs)
2. 15 layers → 13 layers (fine-tune 10 epochs)
3. 13 layers → 12 layers (fine-tune 10 epochs)

---

## 🎯 빠른 시작 (Quick Start)

**가장 간단한 방법으로 바로 시작하기:**

```bash
# 1. Layer Pruning 스크립트 실행
cd /workspace/nemo-main/NeMo-dev

# 2. 모델 축소
python -c "
import nemo.collections.asr as nemo_asr
import torch

# 모델 로드
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('YOUR_TEACHER_MODEL.nemo')

# 12개 레이어 선택 (균등 분포)
keep = [0, 1, 3, 5, 7, 9, 11, 13, 14, 15, 16]
model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in keep])
model.cfg.encoder.n_layers = 12

# 저장
model.save_to('student_12layers.nemo')
print('✓ Student model created!')
"

# 3. Fine-tuning
python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  model.train_ds.manifest_filepath=YOUR_TRAIN_MANIFEST \
  model.validation_ds.manifest_filepath=YOUR_VAL_MANIFEST \
  model.tokenizer.dir=YOUR_TOKENIZER_DIR \
  +init_from_nemo_model=student_12layers.nemo \
  trainer.devices=1 \
  trainer.max_epochs=50
```

---

## 📊 예상 결과

| 모델 | 파라미터 | WER (예상) | 학습 시간 |
|------|----------|------------|-----------|
| Teacher (17 layers) | ~115M | 5.0% | 기준 |
| Student - Pruning | ~60M | 5.5-6.0% | 기준의 20% |
| Student - Distillation | ~60M | 5.2-5.5% | 기준의 100% |
| Student - Progressive | ~60M | 5.3-5.6% | 기준의 150% |

*실제 결과는 데이터셋과 학습 설정에 따라 다를 수 있습니다.

---

## 💡 추가 팁

### 1. 어떤 레이어를 제거할까?

- **중간 레이어 제거**: 일반적으로 가장 안전
- **끝 레이어 제거**: 구현이 간단하지만 성능 저하 가능
- **균등 제거**: 전체적인 구조 유지

### 2. Fine-tuning 하이퍼파라미터

```yaml
# 작은 learning rate
model.optim.lr: 1.0  # 원래 5.0에서 감소

# 짧은 warmup
model.optim.sched.warmup_steps: 1000  # 원래 10000에서 감소

# 적절한 epoch
trainer.max_epochs: 50  # 원래 1000에서 감소
```

### 3. 성능 모니터링

- Validation WER을 주기적으로 확인
- Teacher 모델과 비교
- Overfitting 주의

---

## 🔧 트러블슈팅

### 문제 1: "Shape mismatch" 에러

**원인**: 레이어 수가 맞지 않음

**해결**: 설정 파일의 `n_layers`를 확인

### 문제 2: 성능이 너무 낮음

**해결책:**
1. Fine-tuning epoch 늘리기
2. Learning rate 조정
3. Knowledge Distillation 시도

### 문제 3: OOM (Out of Memory)

**해결책:**
1. Batch size 줄이기
2. `fused_batch_size` 줄이기
3. Gradient accumulation 사용

---

## 📚 참고 자료

- NeMo Documentation: https://docs.nvidia.com/deeplearning/nemo/
- Knowledge Distillation Paper: https://arxiv.org/abs/1503.02531
- FastConformer: https://arxiv.org/abs/2305.05084

---

## ✅ 체크리스트

학습 시작 전 확인사항:

- [ ] Teacher 모델 (.nemo) 준비됨
- [ ] 학습/검증 데이터 manifest 준비됨
- [ ] Tokenizer 준비됨
- [ ] 충분한 GPU 메모리 확인
- [ ] 설정 파일 수정 완료
- [ ] 출력 디렉토리 설정 완료

---

**작성일**: 2025-11-22
**버전**: 1.0
