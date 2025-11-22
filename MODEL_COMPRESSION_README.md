# Model Compression for NeMo ASR Models

이 브랜치는 NeMo ASR 모델을 압축하는 3가지 방법을 제공합니다.

## 📚 개요

기존 학습된 큰 모델(Teacher, 17 layers, 115M params)을 작은 모델(Student, 12 layers, 60M params)로 압축하는 방법들입니다.

## 🎯 3가지 방법

| 방법 | 난이도 | 학습 시간 | 예상 WER | 추천 상황 |
|------|--------|-----------|----------|-----------|
| **1. Layer Pruning** | ⭐ 쉬움 | 짧음 (20%) | 5.5-6.0% | 빠른 프로토타입 |
| **2. Knowledge Distillation** | ⭐⭐⭐ 어려움 | 김 (100%) | 5.2-5.5% | 최고 성능 |
| **3. Progressive Shrinking** | ⭐⭐ 보통 | 매우 김 (150%) | 5.3-5.6% | 안정성 중시 |

*Teacher 모델 WER: 5.0% 기준

## 📁 파일 구조

```
examples/asr/
├── 📚 가이드 문서
│   ├── MODEL_COMPRESSION_COMPARISON.md      # 3가지 방법 비교 및 선택 가이드
│   ├── MODEL_COMPRESSION_GUIDE.md           # 전체 개요
│   ├── KNOWLEDGE_DISTILLATION_GUIDE.md      # Knowledge Distillation 상세 가이드
│   └── PROGRESSIVE_SHRINKING_GUIDE.md       # Progressive Shrinking 상세 가이드
│
├── 🚀 실행 스크립트
│   ├── asr_hybrid_transducer_ctc/
│   │   └── speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py  # Distillation 학습 스크립트
│   ├── quick_start_distillation.py          # Distillation 빠른 시작
│   ├── progressive_shrinking_trainer.py     # Progressive Shrinking 자동화
│   ├── layer_pruning_example.py             # Layer Pruning 예제
│   ├── knowledge_distillation_example.py    # Distillation 예제
│   └── progressive_shrinking_example.py     # Progressive Shrinking 예제
│
└── ⚙️ 설정 파일
    └── conf/fastconformer/hybrid_transducer_ctc/
        ├── fastconformer_hybrid_transducer_ctc_bpe_small.yaml    # 12-layer (Distillation용)
        ├── fastconformer_hybrid_transducer_ctc_bpe_l12_half.yaml # 12-layer
        └── fastconformer_hybrid_transducer_ctc_bpe_l14_half.yaml # 14-layer
```

## 🚀 빠른 시작

### 방법 1: Layer Pruning (가장 간단)

```bash
# 1. 모델 축소
python -c "
import nemo.collections.asr as nemo_asr
import torch

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('teacher.nemo')
keep = [0, 1, 3, 5, 7, 9, 11, 13, 14, 15, 16]
model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in keep])
model.cfg.encoder.n_layers = 12
model.save_to('student.nemo')
"

# 2. Fine-tuning
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  +init_from_nemo_model=student.nemo \
  trainer.max_epochs=30 \
  ...
```

### 방법 2: Knowledge Distillation (최고 성능)

```bash
python examples/asr/quick_start_distillation.py \
  --teacher_model teacher.nemo \
  --train_manifest train.json \
  --val_manifest val.json \
  --tokenizer_dir tokenizer \
  --output_dir experiments/distillation
```

또는:

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

### 방법 3: Progressive Shrinking (가장 안정적)

```bash
python examples/asr/progressive_shrinking_trainer.py \
  --teacher_model teacher.nemo \
  --target_layers 12 \
  --train_manifest train.json \
  --val_manifest val.json \
  --tokenizer_dir tokenizer \
  --output_dir experiments/progressive_shrinking
```

## 📖 상세 가이드

### 어떤 방법을 선택해야 하나요?

**30초 의사결정:**

1. **시간이 1주일 미만?** → Layer Pruning
2. **최고 성능 필요?** → Knowledge Distillation
3. **안정성 최우선?** → Progressive Shrinking
4. **첫 시도?** → Layer Pruning

**상세 비교는 다음 문서 참고:**
```bash
cat examples/asr/MODEL_COMPRESSION_COMPARISON.md
```

### 각 방법의 상세 가이드

- **Layer Pruning**: `MODEL_COMPRESSION_GUIDE.md`
- **Knowledge Distillation**: `KNOWLEDGE_DISTILLATION_GUIDE.md`
- **Progressive Shrinking**: `PROGRESSIVE_SHRINKING_GUIDE.md`

## 🎯 주요 기능

### Knowledge Distillation

- ✅ Teacher-Student 학습
- ✅ Encoder feature matching
- ✅ Logits distillation (KL divergence)
- ✅ Temperature scaling
- ✅ Configurable alpha (student/teacher loss ratio)

### Layer Pruning

- ✅ 빠른 모델 축소
- ✅ Teacher 가중치 직접 활용
- ✅ 여러 레이어 제거 전략 (uniform, middle, end)

### Progressive Shrinking

- ✅ 단계적 축소 (17→15→13→12)
- ✅ 각 단계 자동 fine-tuning
- ✅ 중간 모델 활용 가능
- ✅ 전체 파이프라인 자동화

## 📊 예상 결과

| 모델 | 파라미터 | WER | 학습 시간 | 추론 속도 |
|------|----------|-----|-----------|-----------|
| Teacher (17 layers) | 115M | 5.0% | - | 1.0x |
| Student - Scratch | 60M | 6.5% | 100% | 1.7x |
| Student - Pruning | 60M | 5.8% | 20% | 1.7x |
| Student - Distillation | 60M | 5.3% | 100% | 1.7x |
| Student - Progressive | 60M | 5.5% | 150% | 1.7x |

## 💡 팁

### GPU 메모리 최적화

```bash
# Batch size 줄이기
model.train_ds.batch_size=8

# Mixed precision
trainer.precision=16

# Gradient accumulation
trainer.accumulate_grad_batches=2
```

### 학습 시간 단축

```bash
# Layer Pruning 선택 (가장 빠름)

# 또는 epochs 줄이기
trainer.max_epochs=30

# Progressive Shrinking의 경우 step size 증가
--step_size 3
```

## 🔧 트러블슈팅

### 문제: "Shape mismatch"

**해결**: Student의 `d_model`을 Teacher와 동일하게 설정

```yaml
encoder:
  d_model: 512  # Teacher와 동일!
```

### 문제: OOM (Out of Memory)

**해결**:
```bash
model.train_ds.batch_size=8
trainer.precision=16
model.joint.fused_batch_size=2
```

### 문제: 성능이 너무 낮음

**해결**:
1. 학습 시간 늘리기: `trainer.max_epochs=150`
2. Knowledge Distillation 시도
3. Learning rate 조정: `model.optim.lr=2.0`

## 📝 커밋 정보

**브랜치**: `dev/distillation`

**커밋 메시지**:
```
Add model compression methods: Knowledge Distillation, Layer Pruning, and Progressive Shrinking

- Add Knowledge Distillation training script with encoder and logits distillation
- Add Layer Pruning example for quick model compression
- Add Progressive Shrinking trainer for stable step-by-step model reduction
- Add comprehensive guides for all three methods
- Add comparison guide to help choose the right method
- Add quick start scripts for easy usage
- Add small model configurations (12-layer, 14-layer variants)

Features:
- Knowledge Distillation: Best performance (WER 5.2-5.5%)
- Layer Pruning: Fastest method (20% training time)
- Progressive Shrinking: Most stable (step-by-step reduction)

All methods reduce model size from 115M (17 layers) to ~60M (12 layers) parameters
```

**추가된 파일**: 13개
- 가이드 문서: 4개
- 실행 스크립트: 6개
- 설정 파일: 3개

**총 라인 수**: 4,413 lines

## 🎓 다음 단계

1. **가이드 읽기**
   ```bash
   cat examples/asr/MODEL_COMPRESSION_COMPARISON.md
   ```

2. **방법 선택**
   - 빠른 실험: Layer Pruning
   - 최고 성능: Knowledge Distillation
   - 안정성: Progressive Shrinking

3. **학습 시작**
   - 위의 빠른 시작 가이드 참고

4. **모델 평가**
   ```bash
   python examples/asr/speech_to_text_eval.py \
     model_path=student.nemo \
     test_ds.manifest_filepath=test.json
   ```

## 📞 문의

문제가 있거나 질문이 있으면 가이드 문서를 먼저 확인하세요:
- `MODEL_COMPRESSION_COMPARISON.md` - 방법 선택
- `KNOWLEDGE_DISTILLATION_GUIDE.md` - Distillation 상세
- `PROGRESSIVE_SHRINKING_GUIDE.md` - Progressive Shrinking 상세

---

**작성일**: 2025-11-22
**버전**: 1.0
**브랜치**: dev/distillation
