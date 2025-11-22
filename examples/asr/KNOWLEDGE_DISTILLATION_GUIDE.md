# Knowledge Distillation 완벽 가이드

## 📚 목차
1. [Knowledge Distillation이란?](#knowledge-distillation이란)
2. [왜 사용하나요?](#왜-사용하나요)
3. [단계별 실행 가이드](#단계별-실행-가이드)
4. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
5. [트러블슈팅](#트러블슈팅)

---

## Knowledge Distillation이란?

**Knowledge Distillation**은 큰 모델(Teacher)의 지식을 작은 모델(Student)에게 전달하는 기법입니다.

### 핵심 아이디어

```
Teacher Model (17 layers, 115M params)
    ↓ 지식 전달
Student Model (12 layers, 60M params)
```

### 작동 원리

1. **Teacher 모델**: 이미 학습된 큰 모델 (성능이 좋음)
2. **Student 모델**: 새로 학습할 작은 모델 (빠르고 가벼움)
3. **학습 방법**:
   - Student는 **실제 정답**과 **Teacher의 출력** 둘 다 학습
   - Teacher의 "soft targets"를 통해 더 많은 정보 습득

### Loss 함수

```
Total Loss = α × Student Loss + (1-α) × Distillation Loss

여기서:
- Student Loss: 실제 정답과의 차이 (CTC/RNNT loss)
- Distillation Loss: Teacher 출력과의 차이 (KL divergence)
- α: 가중치 (보통 0.5~0.7)
```

---

## 왜 사용하나요?

### 장점

✅ **최고 성능**: 단순히 작은 모델을 학습하는 것보다 높은 성능
✅ **효율적 학습**: Teacher의 지식을 활용하여 더 빠르게 수렴
✅ **일반화 능력**: Teacher의 soft targets가 regularization 효과

### 성능 비교 (예상)

| 방법 | WER | 학습 시간 |
|------|-----|-----------|
| 작은 모델 처음부터 학습 | 6.5% | 100% |
| Layer Pruning + Fine-tuning | 5.5-6.0% | 20% |
| **Knowledge Distillation** | **5.2-5.5%** | 100% |

*Teacher 모델 WER: 5.0% 기준

---

## 단계별 실행 가이드

### 준비물 체크리스트

- [ ] Teacher 모델 (.nemo 파일) - 17 layers
- [ ] 학습 데이터 manifest
- [ ] 검증 데이터 manifest
- [ ] Tokenizer 디렉토리
- [ ] GPU (최소 1개)

---

### Step 1: Teacher 모델 준비

기존에 학습된 17-layer 모델이 필요합니다.

```bash
# Teacher 모델 경로 확인
ls -lh /path/to/teacher_model_17layers.nemo

# 모델 정보 확인 (Optional)
python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('teacher_model_17layers.nemo')
print(f'Layers: {len(model.encoder.layers)}')
print(f'D-model: {model.encoder.d_model}')
"
```

---

### Step 2: Student 모델 설정 확인

설정 파일이 이미 준비되어 있습니다:

```bash
# 설정 파일 확인
cat examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe_small.yaml
```

**주요 설정:**
- `n_layers: 12` (17에서 감소)
- `d_model: 512` (Teacher와 동일 - 중요!)
- `pred_hidden: 320` (640에서 감소)
- `joint_hidden: 320` (640에서 감소)

---

### Step 3: Distillation 학습 실행

#### 기본 실행

```bash
cd /workspace/nemo-main/NeMo-dev

python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  model.train_ds.manifest_filepath=/path/to/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
  model.tokenizer.dir=/path/to/tokenizer \
  model.tokenizer.type=bpe \
  teacher_model_path=/path/to/teacher_model_17layers.nemo \
  trainer.devices=1 \
  trainer.max_epochs=100 \
  exp_manager.exp_dir=experiments/distillation_run1
```

#### 고급 설정 (추천)

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  model.train_ds.manifest_filepath=/path/to/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
  model.tokenizer.dir=/path/to/tokenizer \
  model.tokenizer.type=bpe \
  teacher_model_path=/path/to/teacher_model_17layers.nemo \
  distillation.alpha=0.6 \
  distillation.temperature=2.0 \
  distillation.use_encoder_distill=true \
  distillation.use_logits_distill=true \
  trainer.devices=1 \
  trainer.max_epochs=100 \
  trainer.precision=16 \
  model.optim.lr=3.0 \
  model.optim.sched.warmup_steps=5000 \
  exp_manager.exp_dir=experiments/distillation_alpha06_temp20 \
  exp_manager.create_wandb_logger=true \
  exp_manager.wandb_logger_kwargs.project=asr_distillation \
  exp_manager.wandb_logger_kwargs.name=student_12layers_distill
```

---

### Step 4: 학습 모니터링

#### TensorBoard로 확인

```bash
tensorboard --logdir experiments/distillation_run1
```

**확인할 지표:**
- `train_loss_total`: 전체 loss
- `train_loss_student`: Student loss (실제 정답과의 차이)
- `train_loss_distill`: Distillation loss (Teacher와의 차이)
- `distill_loss_encoder`: Encoder feature matching loss
- `distill_loss_logits`: Logits KL divergence loss
- `val_wer`: Validation WER (가장 중요!)

#### 로그 확인

```bash
tail -f experiments/distillation_run1/nemo_log_globalrank-0_localrank-0.txt
```

**정상 학습 시 보이는 것:**
```
📚 Knowledge Distillation enabled
   Teacher model: /path/to/teacher_model_17layers.nemo
   Alpha (student weight): 0.6
   Temperature: 2.0
✓ Teacher model loaded and frozen
  Teacher layers: 17
  Student layers: 12
  Distillation alpha: 0.6
  Temperature: 2.0
🚀 Starting distillation training...
```

---

## 하이퍼파라미터 튜닝

### 1. Alpha (α) - Student Loss 가중치

**의미**: Student loss와 Distillation loss의 비율

| Alpha | Student Loss | Distillation Loss | 추천 상황 |
|-------|--------------|-------------------|-----------|
| 0.3 | 30% | 70% | Teacher를 많이 신뢰 |
| **0.5** | **50%** | **50%** | **균형 (기본값)** |
| 0.7 | 70% | 30% | 실제 정답 중시 |

**추천값**: `0.5 ~ 0.6`

```bash
# Alpha 실험
distillation.alpha=0.5  # 균형잡힌 학습
distillation.alpha=0.6  # Student loss 약간 더 중시
distillation.alpha=0.7  # 실제 정답 더 중시
```

---

### 2. Temperature (T) - Soft Targets 정도

**의미**: Teacher의 출력을 얼마나 "soft"하게 만들지

| Temperature | 효과 | 추천 상황 |
|-------------|------|-----------|
| 1.0 | Hard targets (일반 확률) | 사용 안함 |
| **2.0** | **Soft targets (기본값)** | **대부분의 경우** |
| 4.0 | Very soft targets | Teacher가 매우 확실할 때 |

**추천값**: `2.0 ~ 3.0`

```bash
# Temperature 실험
distillation.temperature=2.0  # 기본값
distillation.temperature=3.0  # 더 soft한 targets
```

**원리:**
```python
# Temperature=1.0 (hard)
[0.9, 0.05, 0.03, 0.02]  # 첫 번째가 압도적

# Temperature=2.0 (soft)
[0.6, 0.2, 0.15, 0.05]   # 다른 클래스도 정보 제공
```

---

### 3. Distillation 타입

#### Encoder Distillation (Feature Matching)

```bash
distillation.use_encoder_distill=true  # 추천
```

- Encoder의 출력 벡터를 직접 비교
- MSE loss 사용
- 중간 representation 학습에 효과적

#### Logits Distillation

```bash
distillation.use_logits_distill=true  # 추천
```

- CTC decoder의 logits 비교
- KL divergence 사용
- 최종 출력 분포 학습에 효과적

**추천**: 둘 다 사용 (기본값)

---

### 4. Learning Rate 조정

Distillation 시에는 일반 학습보다 약간 낮은 learning rate 추천:

```bash
# 일반 학습
model.optim.lr=5.0

# Distillation (추천)
model.optim.lr=3.0  # 약간 낮춤

# Warmup도 조정
model.optim.sched.warmup_steps=5000  # 원래 10000에서 감소
```

---

## 실험 예시

### 실험 1: 기본 설정

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=val.json \
  model.tokenizer.dir=tokenizer \
  teacher_model_path=teacher.nemo \
  distillation.alpha=0.5 \
  distillation.temperature=2.0 \
  trainer.devices=1 \
  trainer.max_epochs=100 \
  exp_manager.exp_dir=exp/baseline
```

### 실험 2: Student Loss 중시

```bash
# Alpha를 높여서 실제 정답을 더 중시
distillation.alpha=0.7 \
exp_manager.exp_dir=exp/alpha07
```

### 실험 3: 더 Soft한 Targets

```bash
# Temperature를 높여서 더 많은 정보 전달
distillation.temperature=3.0 \
exp_manager.exp_dir=exp/temp30
```

### 실험 4: Encoder만 Distillation

```bash
# Encoder feature matching만 사용
distillation.use_encoder_distill=true \
distillation.use_logits_distill=false \
exp_manager.exp_dir=exp/encoder_only
```

---

## 트러블슈팅

### 문제 1: "Teacher model not found"

**증상:**
```
FileNotFoundError: teacher_model_17layers.nemo not found
```

**해결:**
```bash
# 경로 확인
ls -lh /path/to/teacher_model_17layers.nemo

# 절대 경로 사용
teacher_model_path=/absolute/path/to/teacher_model_17layers.nemo
```

---

### 문제 2: "Shape mismatch in encoder"

**증상:**
```
RuntimeError: size mismatch, m1: [B, 512], m2: [256, ...]
```

**원인**: Teacher와 Student의 `d_model`이 다름

**해결**: Student의 `d_model`을 Teacher와 동일하게 설정
```yaml
# fastconformer_hybrid_transducer_ctc_bpe_small.yaml
encoder:
  d_model: 512  # Teacher와 동일하게!
```

---

### 문제 3: Distillation Loss가 너무 높음

**증상:**
```
train_loss_distill: 15.3
train_loss_student: 2.1
```

**원인**: Temperature가 너무 낮거나 모델 차이가 큼

**해결:**
```bash
# Temperature 증가
distillation.temperature=3.0

# Alpha 조정 (student loss 더 중시)
distillation.alpha=0.7
```

---

### 문제 4: 성능이 Teacher보다 너무 낮음

**증상:**
```
Teacher WER: 5.0%
Student WER: 8.0%  # 너무 큰 차이
```

**해결 방법:**

1. **학습 시간 늘리기**
```bash
trainer.max_epochs=150  # 100 → 150
```

2. **Alpha 조정**
```bash
distillation.alpha=0.4  # Teacher를 더 신뢰
```

3. **Learning rate 낮추기**
```bash
model.optim.lr=2.0  # 더 천천히 학습
```

4. **Encoder distillation 활성화**
```bash
distillation.use_encoder_distill=true
```

---

### 문제 5: OOM (Out of Memory)

**해결:**
```bash
# Batch size 줄이기
model.train_ds.batch_size=8  # 16 → 8

# Fused batch size 줄이기
model.joint.fused_batch_size=2  # 4 → 2

# Mixed precision 사용
trainer.precision=16  # 32 → 16

# Gradient accumulation
trainer.accumulate_grad_batches=2
```

---

## 성능 최적화 팁

### 1. 단계적 학습 (Curriculum Learning)

```bash
# Stage 1: Alpha 낮게 시작 (Teacher 중시)
distillation.alpha=0.3
trainer.max_epochs=30

# Stage 2: Alpha 증가 (균형)
distillation.alpha=0.5
trainer.max_epochs=50

# Stage 3: Alpha 높게 (Student 중시)
distillation.alpha=0.7
trainer.max_epochs=20
```

### 2. Teacher Ensemble (고급)

여러 Teacher 모델 사용 (구현 필요)

### 3. Layer-wise Distillation (고급)

각 레이어별로 distillation (구현 필요)

---

## 예상 결과

### 학습 곡선

```
Epoch 1:  val_wer=15.0%  (초기)
Epoch 10: val_wer=8.0%   (빠른 수렴)
Epoch 30: val_wer=6.0%
Epoch 50: val_wer=5.5%
Epoch 100: val_wer=5.2%  (최종)
```

### 최종 성능 비교

| 모델 | 파라미터 | WER | 추론 속도 |
|------|----------|-----|-----------|
| Teacher (17 layers) | 115M | 5.0% | 1.0x |
| Student - Scratch | 60M | 6.5% | 1.7x |
| Student - Pruning | 60M | 5.8% | 1.7x |
| **Student - Distillation** | **60M** | **5.2%** | **1.7x** |

---

## 체크리스트

학습 전:
- [ ] Teacher 모델 준비됨
- [ ] Student 설정 파일 확인
- [ ] 데이터 manifest 준비
- [ ] Tokenizer 준비
- [ ] GPU 메모리 충분 (최소 16GB)

학습 중:
- [ ] TensorBoard 모니터링 중
- [ ] val_wer 감소 확인
- [ ] distill_loss 안정적
- [ ] 로그 정상

학습 후:
- [ ] 최종 WER 확인
- [ ] Teacher와 비교
- [ ] 모델 저장 확인
- [ ] 추론 속도 테스트

---

## 다음 단계

1. **모델 평가**
```bash
python examples/asr/speech_to_text_eval.py \
  model_path=experiments/distillation_run1/checkpoints/best.nemo \
  test_ds.manifest_filepath=test.json
```

2. **모델 내보내기**
```bash
# ONNX export
python examples/asr/export/export_rnnt.py \
  model_path=best.nemo \
  output_path=student_model.onnx
```

3. **추론 테스트**
```bash
python examples/asr/transcribe_speech.py \
  model_path=best.nemo \
  audio_dir=/path/to/audio/files
```

---

**작성일**: 2025-11-22
**버전**: 1.0
**난이도**: ⭐⭐⭐ 중급~고급
