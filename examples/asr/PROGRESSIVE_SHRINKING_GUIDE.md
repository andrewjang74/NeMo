# Progressive Shrinking 완벽 가이드

## 📚 목차
1. [Progressive Shrinking이란?](#progressive-shrinking이란)
2. [왜 사용하나요?](#왜-사용하나요)
3. [단계별 실행 가이드](#단계별-실행-가이드)
4. [레이어 제거 전략](#레이어-제거-전략)
5. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
6. [트러블슈팅](#트러블슈팅)

---

## Progressive Shrinking이란?

**Progressive Shrinking**은 큰 모델을 **단계적으로 줄여가면서** 각 단계마다 fine-tuning하는 방법입니다.

### 핵심 아이디어

```
Step 1: 17 layers → 15 layers (fine-tune 10 epochs)
           ↓
Step 2: 15 layers → 13 layers (fine-tune 10 epochs)
           ↓
Step 3: 13 layers → 12 layers (fine-tune 10 epochs)
           ↓
      Final: 12 layers (완성!)
```

### 작동 원리

1. **점진적 축소**: 한 번에 많이 줄이지 않고 조금씩 줄임
2. **각 단계 fine-tuning**: 축소 후 바로 학습하여 성능 회복
3. **안정적 학습**: 급격한 변화를 피하여 안정성 확보

---

## 왜 사용하나요?

### 장점

✅ **안정적**: 급격한 모델 변화를 피함
✅ **단계별 확인**: 각 단계마다 성능 확인 가능
✅ **점진적 적응**: 모델이 서서히 작아지는 것에 적응
✅ **중간 모델 활용**: 각 단계의 모델도 사용 가능 (15-layer, 13-layer 등)

### 단점

❌ **학습 시간 김**: 여러 단계를 거쳐야 함
❌ **관리 복잡**: 여러 중간 모델 관리 필요

---

## 방법 비교

| 방법 | 안정성 | 학습 시간 | 성능 | 추천도 |
|------|--------|-----------|------|--------|
| Layer Pruning | 중간 | 짧음 (20%) | 중간 | ⭐⭐⭐⭐ |
| Knowledge Distillation | 중간 | 김 (100%) | 높음 | ⭐⭐⭐⭐⭐ |
| **Progressive Shrinking** | **높음** | **매우 김 (150%)** | **높음** | **⭐⭐⭐** |

**추천 상황:**
- 안정성이 가장 중요한 경우
- 학습 시간이 충분한 경우
- 중간 크기 모델도 필요한 경우

---

## 단계별 실행 가이드

### 준비물 체크리스트

- [ ] Teacher 모델 (.nemo 파일) - 17 layers
- [ ] 학습 데이터 manifest
- [ ] 검증 데이터 manifest
- [ ] Tokenizer 디렉토리
- [ ] GPU (최소 1개)
- [ ] 충분한 시간 (일반 학습의 1.5배)

---

### 방법 A: 자동화 스크립트 사용 (추천) ⭐

가장 쉽고 안전한 방법입니다.

```bash
cd /workspace/nemo-main/NeMo-dev/examples/asr

python progressive_shrinking_trainer.py \
  --teacher_model /path/to/teacher_17layers.nemo \
  --target_layers 12 \
  --train_manifest /path/to/train.json \
  --val_manifest /path/to/val.json \
  --tokenizer_dir /path/to/tokenizer \
  --output_dir experiments/progressive_shrinking
```

**이 스크립트가 자동으로:**
1. ✅ 단계 계획 수립 (17→15→13→12)
2. ✅ 각 단계마다 모델 축소
3. ✅ 각 단계마다 fine-tuning
4. ✅ 로그 및 체크포인트 관리
5. ✅ 최종 결과 요약

---

### 방법 B: 수동으로 단계별 실행

각 단계를 직접 제어하고 싶을 때 사용합니다.

#### Step 1: 17 layers → 15 layers

```python
# 1-1. 모델 축소
import nemo.collections.asr as nemo_asr
import torch

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    'teacher_17layers.nemo'
)

# 균등하게 15개 레이어 선택
keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]  # 15개
model.encoder.layers = torch.nn.ModuleList([
    model.encoder.layers[i] for i in keep_indices
])
model.cfg.encoder.n_layers = 15

model.save_to('step1_15layers.nemo')
```

```bash
# 1-2. Fine-tuning
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=val.json \
  model.tokenizer.dir=tokenizer \
  +init_from_nemo_model=step1_15layers.nemo \
  trainer.devices=1 \
  trainer.max_epochs=10 \
  model.optim.lr=1.0 \
  exp_manager.exp_dir=experiments/step1_15layers
```

#### Step 2: 15 layers → 13 layers

```python
# 2-1. 모델 축소
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    'experiments/step1_15layers/checkpoints/best.nemo'
)

# 13개 레이어 선택
keep_indices = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]  # 13개
model.encoder.layers = torch.nn.ModuleList([
    model.encoder.layers[i] for i in keep_indices
])
model.cfg.encoder.n_layers = 13

model.save_to('step2_13layers.nemo')
```

```bash
# 2-2. Fine-tuning
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=val.json \
  model.tokenizer.dir=tokenizer \
  +init_from_nemo_model=step2_13layers.nemo \
  trainer.devices=1 \
  trainer.max_epochs=10 \
  model.optim.lr=1.0 \
  exp_manager.exp_dir=experiments/step2_13layers
```

#### Step 3: 13 layers → 12 layers

```python
# 3-1. 모델 축소
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    'experiments/step2_13layers/checkpoints/best.nemo'
)

# 12개 레이어 선택
keep_indices = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12]  # 12개
model.encoder.layers = torch.nn.ModuleList([
    model.encoder.layers[i] for i in keep_indices
])
model.cfg.encoder.n_layers = 12

model.save_to('step3_12layers.nemo')
```

```bash
# 3-2. Fine-tuning
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=val.json \
  model.tokenizer.dir=tokenizer \
  +init_from_nemo_model=step3_12layers.nemo \
  trainer.devices=1 \
  trainer.max_epochs=10 \
  model.optim.lr=1.0 \
  exp_manager.exp_dir=experiments/step3_12layers
```

---

## 레이어 제거 전략

### 1. Uniform (균등 분포) - 추천 ⭐

전체 레이어를 균등하게 샘플링합니다.

```python
# 17 → 12 layers
original = 17
target = 12
step = original / target
keep_indices = [int(i * step) for i in range(target)]
# 결과: [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16]
```

**장점:**
- ✅ 전체 구조를 균등하게 유지
- ✅ 초기/중간/후기 레이어 모두 포함
- ✅ 가장 안정적

**사용:**
```bash
--remove_strategy uniform
```

---

### 2. Middle (중간 제거)

처음과 끝 레이어를 유지하고 중간을 제거합니다.

```python
# 17 → 12 layers (5개 제거)
layers_to_remove = 5
remove_start = 17 // 2 - 5 // 2  # 중간 시작점
keep_indices = list(range(remove_start)) + list(range(remove_start + 5, 17))
# 결과: [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]
```

**장점:**
- ✅ 초기 feature extraction 유지
- ✅ 최종 출력 레이어 유지

**사용:**
```bash
--remove_strategy middle
```

---

### 3. End (끝 제거)

끝 레이어를 제거합니다.

```python
# 17 → 12 layers
keep_indices = list(range(12))
# 결과: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

**장점:**
- ✅ 구현이 가장 간단

**단점:**
- ❌ 고수준 representation 손실 가능

**사용:**
```bash
--remove_strategy end
```

---

## 하이퍼파라미터 튜닝

### 1. Step Size (단계 크기)

각 단계마다 줄일 레이어 수

| Step Size | 단계 수 | 총 Epochs | 추천 상황 |
|-----------|---------|-----------|-----------|
| 1 | 5 (17→16→15→14→13→12) | 50 | 매우 안정적, 시간 많음 |
| **2** | **3 (17→15→13→12)** | **30** | **균형 (기본값)** |
| 3 | 2 (17→14→12) | 20 | 빠르게, 시간 부족 |
| 5 | 1 (17→12) | 10 | Layer Pruning과 동일 |

```bash
--step_size 2  # 기본값
```

---

### 2. Epochs Per Step (각 단계 Epoch)

각 단계의 fine-tuning epoch 수

| Epochs | 학습 시간 | 성능 | 추천 상황 |
|--------|-----------|------|-----------|
| 5 | 짧음 | 낮음 | 빠른 실험 |
| **10** | **중간** | **중간** | **기본값** |
| 20 | 김 | 높음 | 최고 성능 |
| 30 | 매우 김 | 매우 높음 | 프로덕션 |

```bash
--epochs_per_step 10  # 기본값
```

---

### 3. Learning Rate

Fine-tuning 시 learning rate

| LR | 효과 | 추천 상황 |
|----|------|-----------|
| 0.5 | 매우 안정적 | 초기 실험 |
| **1.0** | **안정적 (기본값)** | **대부분의 경우** |
| 2.0 | 빠른 수렴 | 시간 부족 |
| 3.0 | 불안정 가능 | 주의 필요 |

```bash
--learning_rate 1.0  # 기본값
```

---

## 실험 예시

### 실험 1: 기본 설정 (추천)

```bash
python progressive_shrinking_trainer.py \
  --teacher_model teacher_17layers.nemo \
  --target_layers 12 \
  --train_manifest train.json \
  --val_manifest val.json \
  --tokenizer_dir tokenizer \
  --output_dir experiments/progressive_basic \
  --step_size 2 \
  --epochs_per_step 10 \
  --remove_strategy uniform
```

**예상 시간**: 일반 학습의 1.5배
**예상 성능**: WER 5.3-5.6%

---

### 실험 2: 빠른 버전

```bash
python progressive_shrinking_trainer.py \
  --teacher_model teacher_17layers.nemo \
  --target_layers 12 \
  --train_manifest train.json \
  --val_manifest val.json \
  --tokenizer_dir tokenizer \
  --output_dir experiments/progressive_fast \
  --step_size 3 \
  --epochs_per_step 5 \
  --remove_strategy uniform
```

**예상 시간**: 일반 학습의 0.5배
**예상 성능**: WER 5.5-6.0%

---

### 실험 3: 최고 성능

```bash
python progressive_shrinking_trainer.py \
  --teacher_model teacher_17layers.nemo \
  --target_layers 12 \
  --train_manifest train.json \
  --val_manifest val.json \
  --tokenizer_dir tokenizer \
  --output_dir experiments/progressive_best \
  --step_size 1 \
  --epochs_per_step 20 \
  --remove_strategy uniform \
  --learning_rate 0.5
```

**예상 시간**: 일반 학습의 2배
**예상 성능**: WER 5.2-5.4%

---

## 학습 모니터링

### 로그 확인

```bash
# 실시간 로그
tail -f experiments/progressive_shrinking/progressive_shrinking.log

# 요약 확인
cat experiments/progressive_shrinking/summary.json
```

### TensorBoard

각 단계별로 TensorBoard 확인:

```bash
# Step 1
tensorboard --logdir experiments/progressive_shrinking/step1_finetune_15layers

# Step 2
tensorboard --logdir experiments/progressive_shrinking/step2_finetune_13layers

# Step 3
tensorboard --logdir experiments/progressive_shrinking/step3_finetune_12layers
```

### 성능 추적

각 단계의 WER을 추적:

```
Step 1 (15 layers):
  Initial WER: 7.0%
  After fine-tune: 5.3%

Step 2 (13 layers):
  Initial WER: 6.5%
  After fine-tune: 5.4%

Step 3 (12 layers):
  Initial WER: 6.2%
  After fine-tune: 5.5%
```

---

## 트러블슈팅

### 문제 1: 중간 단계에서 성능 급락

**증상:**
```
Step 1: WER 5.3% ✓
Step 2: WER 8.5% ❌ (급락!)
```

**원인**: 한 번에 너무 많은 레이어 제거

**해결:**
```bash
# Step size 줄이기
--step_size 1  # 2 → 1

# Epochs 늘리기
--epochs_per_step 20  # 10 → 20
```

---

### 문제 2: Fine-tuning이 수렴하지 않음

**증상:**
```
Epoch 1: loss=3.5
Epoch 5: loss=3.4
Epoch 10: loss=3.4  # 변화 없음
```

**해결:**
```bash
# Learning rate 조정
--learning_rate 2.0  # 1.0 → 2.0

# Epochs 늘리기
--epochs_per_step 15
```

---

### 문제 3: 학습 시간이 너무 김

**해결:**

1. **Step size 증가**
```bash
--step_size 3  # 단계 수 줄이기
```

2. **Epochs 감소**
```bash
--epochs_per_step 5  # 빠른 fine-tuning
```

3. **Mixed precision**
```bash
--precision 16  # 32 → 16
```

4. **Batch size 증가** (GPU 메모리 허용 시)
```bash
--batch_size 32  # 16 → 32
```

---

### 문제 4: 특정 단계에서 실패

**증상:**
```
Step 1: ✓ 완료
Step 2: ❌ 학습 중 오류
```

**해결:**

1. **해당 단계부터 재시작**
```python
# Step 2부터 수동 실행
python progressive_shrinking_trainer.py \
  --teacher_model experiments/progressive_shrinking/step1_finetune_15layers/checkpoints/best.nemo \
  --target_layers 12 \
  ...
```

2. **로그 확인**
```bash
tail -100 experiments/progressive_shrinking/step2_finetune_13layers/nemo_log_*.txt
```

---

## 성능 비교

### 예상 결과

| 단계 | 레이어 | 초기 WER | Fine-tune 후 WER |
|------|--------|----------|------------------|
| Teacher | 17 | - | 5.0% |
| Step 1 | 15 | 6.5% | 5.3% |
| Step 2 | 13 | 6.0% | 5.4% |
| **Step 3** | **12** | **5.8%** | **5.5%** |

### 방법별 비교

| 방법 | 최종 WER | 학습 시간 | 안정성 |
|------|----------|-----------|--------|
| Layer Pruning | 5.8% | 20% | 중간 |
| Knowledge Distillation | 5.2% | 100% | 중간 |
| **Progressive Shrinking** | **5.5%** | **150%** | **높음** |

---

## 장단점 요약

### 장점

✅ **매우 안정적**: 급격한 변화 없음
✅ **단계별 확인**: 각 단계 성능 모니터링
✅ **중간 모델 활용**: 15-layer, 13-layer 모델도 사용 가능
✅ **점진적 적응**: 모델이 서서히 적응

### 단점

❌ **학습 시간 김**: 여러 단계 필요
❌ **관리 복잡**: 많은 중간 모델
❌ **디스크 공간**: 각 단계 체크포인트 저장

---

## 언제 사용하나요?

### Progressive Shrinking 추천:

1. ✅ **안정성이 최우선**
   - 프로덕션 배포
   - 성능 보장 필요

2. ✅ **학습 시간 충분**
   - 긴 학습 시간 가능
   - 리소스 충분

3. ✅ **중간 모델 필요**
   - 여러 크기 모델 필요
   - A/B 테스트

### 다른 방법 추천:

- **빠른 프로토타이핑**: Layer Pruning
- **최고 성능**: Knowledge Distillation
- **균형**: Layer Pruning + 긴 fine-tuning

---

## 다음 단계

### 1. 모델 평가

```bash
# 각 단계 모델 평가
python examples/asr/speech_to_text_eval.py \
  model_path=experiments/progressive_shrinking/step1_finetune_15layers/checkpoints/best.nemo \
  test_ds.manifest_filepath=test.json

python examples/asr/speech_to_text_eval.py \
  model_path=experiments/progressive_shrinking/step3_finetune_12layers/checkpoints/best.nemo \
  test_ds.manifest_filepath=test.json
```

### 2. 최적 모델 선택

성능과 크기의 균형을 고려하여 선택:
- 15-layer: 더 높은 성능
- 13-layer: 균형
- 12-layer: 가장 작음

### 3. 추가 Fine-tuning (Optional)

최종 모델을 더 오래 학습:

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  +init_from_nemo_model=final_12layers.nemo \
  trainer.max_epochs=50 \
  ...
```

---

## 체크리스트

### 학습 전
- [ ] Teacher 모델 준비
- [ ] 데이터 manifest 준비
- [ ] Tokenizer 준비
- [ ] 충분한 디스크 공간 (각 단계 체크포인트)
- [ ] 충분한 시간 (일반 학습의 1.5배)

### 학습 중
- [ ] 각 단계 로그 확인
- [ ] 각 단계 WER 추적
- [ ] 성능 급락 없는지 확인

### 학습 후
- [ ] 모든 단계 완료 확인
- [ ] 최종 모델 성능 확인
- [ ] 중간 모델 성능 비교
- [ ] 최적 모델 선택

---

**Progressive Shrinking**은 가장 안정적인 방법입니다!

- ✅ 단계적으로 안전하게 축소
- ✅ 각 단계마다 성능 확인
- ✅ 여러 크기의 모델 확보

시간이 충분하다면 이 방법을 추천합니다! 🚀

**작성일**: 2025-11-22
**버전**: 1.0
**난이도**: ⭐⭐ 중급
