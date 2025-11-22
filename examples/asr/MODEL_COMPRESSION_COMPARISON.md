# 모델 압축 방법 완벽 비교 가이드

## 📚 목차
1. [3가지 방법 한눈에 비교](#3가지-방법-한눈에-비교)
2. [상황별 추천](#상황별-추천)
3. [의사결정 플로우차트](#의사결정-플로우차트)
4. [실전 예시](#실전-예시)
5. [FAQ](#faq)

---

## 3가지 방법 한눈에 비교

### 방법 요약

| 방법 | 핵심 아이디어 | 난이도 | 학습 시간 | 예상 성능 |
|------|--------------|--------|-----------|-----------|
| **1. Layer Pruning** | 레이어 제거 + Fine-tuning | ⭐ 쉬움 | 짧음 (20%) | WER 5.5-6.0% |
| **2. Knowledge Distillation** | Teacher 지식 전달 | ⭐⭐⭐ 어려움 | 김 (100%) | WER 5.2-5.5% |
| **3. Progressive Shrinking** | 단계적 축소 + Fine-tuning | ⭐⭐ 보통 | 매우 김 (150%) | WER 5.3-5.6% |

*Teacher 모델 WER: 5.0% 기준

---

## 상세 비교표

### 성능 비교

| 항목 | Layer Pruning | Knowledge Distillation | Progressive Shrinking |
|------|---------------|------------------------|----------------------|
| **최종 WER** | 5.5-6.0% | **5.2-5.5%** ⭐ | 5.3-5.6% |
| **Teacher 대비** | +0.5-1.0% | **+0.2-0.5%** ⭐ | +0.3-0.6% |
| **안정성** | 중간 | 중간 | **높음** ⭐ |
| **재현성** | 높음 | 중간 | 높음 |

---

### 구현 난이도

| 항목 | Layer Pruning | Knowledge Distillation | Progressive Shrinking |
|------|---------------|------------------------|----------------------|
| **코드 복잡도** | **낮음** ⭐ | 높음 | 중간 |
| **설정 복잡도** | **낮음** ⭐ | 높음 | 중간 |
| **디버깅** | **쉬움** ⭐ | 어려움 | 중간 |
| **하이퍼파라미터** | 적음 | **많음** | 중간 |

---

### 리소스 요구사항

| 항목 | Layer Pruning | Knowledge Distillation | Progressive Shrinking |
|------|---------------|------------------------|----------------------|
| **학습 시간** | **20%** ⭐ | 100% | 150% |
| **GPU 메모리** | 중간 | **높음** (Teacher + Student) | 중간 |
| **디스크 공간** | **적음** ⭐ | 중간 | 많음 (중간 모델들) |
| **관리 복잡도** | **낮음** ⭐ | 중간 | 높음 |

---

### 장단점 비교

#### 1. Layer Pruning

**장점:**
- ✅ 가장 간단하고 빠름
- ✅ Teacher 가중치 직접 활용
- ✅ 적은 리소스
- ✅ 빠른 프로토타이핑

**단점:**
- ❌ 성능이 다른 방법보다 낮을 수 있음
- ❌ 한 번에 많이 줄이면 성능 급락

**추천 상황:**
- 빠른 실험이 필요할 때
- 리소스가 제한적일 때
- 첫 번째 시도로 적합

---

#### 2. Knowledge Distillation

**장점:**
- ✅ **최고 성능** ⭐
- ✅ Teacher의 지식 효과적 전달
- ✅ 학계에서 검증된 방법
- ✅ Soft targets로 더 많은 정보

**단점:**
- ❌ 구현이 복잡
- ❌ 하이퍼파라미터 튜닝 필요
- ❌ GPU 메모리 많이 필요 (Teacher + Student)
- ❌ 학습 시간 김

**추천 상황:**
- 최고 성능이 필요할 때
- 프로덕션 배포용
- 충분한 리소스가 있을 때

---

#### 3. Progressive Shrinking

**장점:**
- ✅ **가장 안정적** ⭐
- ✅ 단계별 성능 확인 가능
- ✅ 중간 크기 모델도 활용 가능
- ✅ 점진적 적응

**단점:**
- ❌ 학습 시간이 가장 김
- ❌ 여러 중간 모델 관리 필요
- ❌ 디스크 공간 많이 필요

**추천 상황:**
- 안정성이 최우선일 때
- 학습 시간이 충분할 때
- 여러 크기 모델이 필요할 때

---

## 상황별 추천

### 시나리오 1: "빠르게 프로토타입을 만들고 싶어요"

**추천: Layer Pruning** ⭐

```bash
# 가장 빠른 방법
python -c "
import nemo.collections.asr as nemo_asr
import torch

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('teacher.nemo')
keep = [0, 1, 3, 5, 7, 9, 11, 13, 14, 15, 16]
model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in keep])
model.cfg.encoder.n_layers = 12
model.save_to('student.nemo')
"

# Fine-tuning (짧게)
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  +init_from_nemo_model=student.nemo \
  trainer.max_epochs=30 \
  ...
```

**예상 시간**: 1-2일
**예상 성능**: WER 5.8%

---

### 시나리오 2: "최고 성능이 필요해요 (프로덕션)"

**추천: Knowledge Distillation** ⭐

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
  --config-path=../conf/fastconformer/hybrid_transducer_ctc \
  --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
  teacher_model_path=teacher.nemo \
  distillation.alpha=0.5 \
  distillation.temperature=2.0 \
  trainer.max_epochs=100 \
  ...
```

**예상 시간**: 7-10일
**예상 성능**: WER 5.3%

---

### 시나리오 3: "안정적으로 진행하고 싶어요"

**추천: Progressive Shrinking** ⭐

```bash
python examples/asr/progressive_shrinking_trainer.py \
  --teacher_model teacher.nemo \
  --target_layers 12 \
  --step_size 2 \
  --epochs_per_step 10 \
  ...
```

**예상 시간**: 10-15일
**예상 성능**: WER 5.5%

---

### 시나리오 4: "리소스가 제한적이에요"

**추천: Layer Pruning** ⭐

- GPU 메모리: 중간
- 디스크 공간: 적음
- 학습 시간: 짧음

---

### 시나리오 5: "여러 크기 모델이 필요해요"

**추천: Progressive Shrinking** ⭐

단계별로 다양한 크기 모델 확보:
- 15-layer 모델
- 13-layer 모델
- 12-layer 모델

---

## 의사결정 플로우차트

```
시작
  │
  ├─ 시간이 부족한가?
  │   └─ YES → Layer Pruning ⭐
  │
  ├─ 최고 성능이 필요한가?
  │   └─ YES → Knowledge Distillation ⭐
  │
  ├─ 안정성이 최우선인가?
  │   └─ YES → Progressive Shrinking ⭐
  │
  ├─ GPU 메모리가 부족한가?
  │   └─ YES → Layer Pruning 또는 Progressive Shrinking
  │
  ├─ 여러 크기 모델이 필요한가?
  │   └─ YES → Progressive Shrinking ⭐
  │
  └─ 첫 시도인가?
      └─ YES → Layer Pruning ⭐
```

---

## 실전 예시

### 예시 1: 스타트업 - 빠른 MVP

**상황:**
- 시간: 1주일
- 리소스: GPU 1개
- 목표: 빠른 검증

**선택: Layer Pruning**

```bash
# Day 1: 모델 축소
python layer_pruning_example.py

# Day 2-7: Fine-tuning
python speech_to_text_hybrid_rnnt_ctc_bpe.py \
  +init_from_nemo_model=student.nemo \
  trainer.max_epochs=30
```

**결과:**
- WER: 5.8%
- 배포 가능한 모델 확보

---

### 예시 2: 대기업 - 프로덕션 배포

**상황:**
- 시간: 2주
- 리소스: GPU 4개
- 목표: 최고 성능

**선택: Knowledge Distillation**

```bash
# Week 1-2: Distillation 학습
python speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
  teacher_model_path=teacher.nemo \
  distillation.alpha=0.5 \
  trainer.devices=4 \
  trainer.max_epochs=100
```

**결과:**
- WER: 5.3%
- Teacher 대비 0.3% 차이만

---

### 예시 3: 연구소 - 안정적 실험

**상황:**
- 시간: 3주
- 리소스: GPU 2개
- 목표: 안정적 결과

**선택: Progressive Shrinking**

```bash
# Week 1-3: 단계적 축소
python progressive_shrinking_trainer.py \
  --teacher_model teacher.nemo \
  --target_layers 12 \
  --step_size 2 \
  --epochs_per_step 10
```

**결과:**
- WER: 5.5%
- 15-layer, 13-layer 모델도 확보
- 각 단계 성능 분석 가능

---

## 조합 전략

### 전략 1: Layer Pruning → Knowledge Distillation

1. **Phase 1**: Layer Pruning으로 빠르게 시작
2. **Phase 2**: 결과가 좋으면 Distillation으로 성능 향상

**장점:**
- 빠른 초기 결과
- 나중에 성능 개선

---

### 전략 2: Progressive Shrinking → Fine-tuning

1. **Phase 1**: Progressive Shrinking으로 12-layer 모델 생성
2. **Phase 2**: 추가로 긴 fine-tuning

**장점:**
- 안정적 시작
- 최종 성능 극대화

---

## FAQ

### Q1: 어떤 방법이 가장 좋나요?

**A**: 상황에 따라 다릅니다.

- **빠른 실험**: Layer Pruning
- **최고 성능**: Knowledge Distillation
- **안정성**: Progressive Shrinking

---

### Q2: 처음 시도하는데 어떤 방법을 추천하나요?

**A**: **Layer Pruning**을 추천합니다.

이유:
- 가장 간단
- 빠른 결과
- 적은 리소스
- 다른 방법의 baseline

---

### Q3: 성능 차이가 얼마나 나나요?

**A**: 예상 성능 (Teacher WER 5.0% 기준):

| 방법 | WER | Teacher 대비 |
|------|-----|--------------|
| Layer Pruning | 5.8% | +0.8% |
| Knowledge Distillation | 5.3% | +0.3% |
| Progressive Shrinking | 5.5% | +0.5% |

---

### Q4: 여러 방법을 조합할 수 있나요?

**A**: 네, 가능합니다!

예시:
1. Layer Pruning으로 초기 모델 생성
2. Knowledge Distillation으로 성능 향상
3. Progressive Shrinking의 중간 모델 활용

---

### Q5: GPU 메모리가 부족한데 어떻게 하나요?

**A**: 

1. **Layer Pruning 또는 Progressive Shrinking** 선택
   - Knowledge Distillation은 Teacher + Student 동시 로드 필요

2. **Batch size 줄이기**
```bash
model.train_ds.batch_size=8
```

3. **Mixed precision 사용**
```bash
trainer.precision=16
```

---

### Q6: 학습 시간을 줄이려면?

**A**:

1. **Layer Pruning 선택** (가장 빠름)

2. **Epochs 줄이기**
```bash
trainer.max_epochs=30  # 100 → 30
```

3. **Progressive Shrinking의 step size 증가**
```bash
--step_size 3  # 2 → 3
```

---

### Q7: 중간 크기 모델도 필요한데?

**A**: **Progressive Shrinking** 추천

자동으로 생성되는 모델:
- 15-layer 모델
- 13-layer 모델
- 12-layer 모델

각각 다른 성능/속도 트레이드오프

---

## 빠른 의사결정 가이드

### 30초 안에 결정하기

**질문 1**: 시간이 1주일 미만인가?
- YES → **Layer Pruning**
- NO → 질문 2로

**질문 2**: 최고 성능이 필요한가?
- YES → **Knowledge Distillation**
- NO → 질문 3으로

**질문 3**: 안정성이 최우선인가?
- YES → **Progressive Shrinking**
- NO → **Layer Pruning** (기본값)

---

## 체크리스트

### Layer Pruning 선택 시
- [ ] Teacher 모델 준비
- [ ] 데이터 준비
- [ ] GPU 1개 이상
- [ ] 1-2주 시간

### Knowledge Distillation 선택 시
- [ ] Teacher 모델 준비
- [ ] 데이터 준비
- [ ] GPU 메모리 충분 (16GB+)
- [ ] 2-3주 시간
- [ ] 하이퍼파라미터 튜닝 계획

### Progressive Shrinking 선택 시
- [ ] Teacher 모델 준비
- [ ] 데이터 준비
- [ ] 디스크 공간 충분
- [ ] 3-4주 시간
- [ ] 중간 모델 관리 계획

---

## 최종 추천

### 🥇 첫 시도: Layer Pruning

가장 간단하고 빠른 방법으로 시작하세요.

```bash
python layer_pruning_example.py
```

### 🥈 프로덕션: Knowledge Distillation

최고 성능이 필요하면 이 방법을 사용하세요.

```bash
python speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py
```

### 🥉 안정성: Progressive Shrinking

안정적인 결과가 필요하면 이 방법을 사용하세요.

```bash
python progressive_shrinking_trainer.py
```

---

## 참고 자료

- **Layer Pruning 가이드**: `MODEL_COMPRESSION_GUIDE.md`
- **Knowledge Distillation 가이드**: `KNOWLEDGE_DISTILLATION_GUIDE.md`
- **Progressive Shrinking 가이드**: `PROGRESSIVE_SHRINKING_GUIDE.md`

---

**작성일**: 2025-11-22
**버전**: 1.0

모든 방법이 준비되어 있습니다. 상황에 맞게 선택하세요! 🚀
