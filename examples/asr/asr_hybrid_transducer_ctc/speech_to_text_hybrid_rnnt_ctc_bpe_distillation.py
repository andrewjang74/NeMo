# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Knowledge Distillation Training Script for Hybrid RNNT-CTC Models

이 스크립트는 큰 Teacher 모델의 지식을 작은 Student 모델에 전달합니다.

# 사용 방법:

```sh
python speech_to_text_hybrid_rnnt_ctc_bpe_distillation.py \
    --config-path=../conf/fastconformer/hybrid_transducer_ctc \
    --config-name=fastconformer_hybrid_transducer_ctc_bpe_small \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val manifest> \
    model.tokenizer.dir=<path to tokenizer directory> \
    model.tokenizer.type=bpe \
    teacher_model_path=<path to teacher .nemo model> \
    distillation.alpha=0.5 \
    distillation.temperature=2.0 \
    distillation.use_encoder_distill=true \
    distillation.use_logits_distill=true \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    exp_manager.exp_dir=experiments/distillation
```

# 주요 파라미터:
- teacher_model_path: Teacher 모델 경로 (.nemo)
- distillation.alpha: Student loss 가중치 (0.5 = 50% student, 50% teacher)
- distillation.temperature: Softmax temperature (높을수록 soft targets)
- distillation.use_encoder_distill: Encoder 출력도 distillation 할지 여부
- distillation.use_logits_distill: Logits distillation 사용 여부
"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


class DistillationHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCBPEModel):
    """
    Knowledge Distillation을 지원하는 Hybrid RNNT-CTC 모델
    
    Teacher 모델의 출력을 활용하여 Student 모델을 학습합니다.
    """
    
    def __init__(self, cfg: DictConfig, teacher_model_path: str = None, trainer: pl.Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        # Distillation 설정
        self.distillation_cfg = cfg.get('distillation', {})
        self.alpha = self.distillation_cfg.get('alpha', 0.5)
        self.temperature = self.distillation_cfg.get('temperature', 2.0)
        self.use_encoder_distill = self.distillation_cfg.get('use_encoder_distill', True)
        self.use_logits_distill = self.distillation_cfg.get('use_logits_distill', True)
        
        # Teacher 모델 로드
        self.teacher_model = None
        if teacher_model_path:
            logging.info(f"Loading teacher model from {teacher_model_path}")
            self.teacher_model = EncDecHybridRNNTCTCBPEModel.restore_from(teacher_model_path)
            self.teacher_model.eval()
            
            # Teacher 파라미터 고정
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            logging.info(f"✓ Teacher model loaded and frozen")
            logging.info(f"  Teacher layers: {len(self.teacher_model.encoder.layers)}")
            logging.info(f"  Student layers: {len(self.encoder.layers)}")
            logging.info(f"  Distillation alpha: {self.alpha}")
            logging.info(f"  Temperature: {self.temperature}")
    
    def training_step(self, batch, batch_idx):
        """
        Training step with knowledge distillation
        """
        # Student의 원래 loss 계산
        student_loss_dict = super().training_step(batch, batch_idx)
        
        if self.teacher_model is None:
            # Teacher가 없으면 일반 학습
            return student_loss_dict
        
        # Teacher 모델로부터 출력 얻기
        with torch.no_grad():
            teacher_outputs = self._get_teacher_outputs(batch)
        
        # Student 모델 출력 얻기
        student_outputs = self._get_student_outputs(batch)
        
        # Distillation loss 계산
        distill_loss = self._compute_distillation_loss(
            student_outputs, 
            teacher_outputs
        )
        
        # Combined loss
        if isinstance(student_loss_dict, dict):
            student_loss = student_loss_dict.get('loss', student_loss_dict.get('train_loss', 0))
        else:
            student_loss = student_loss_dict
        
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        
        # Logging
        self.log('train_loss_student', student_loss, prog_bar=False, sync_dist=True)
        self.log('train_loss_distill', distill_loss, prog_bar=False, sync_dist=True)
        self.log('train_loss_total', total_loss, prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def _get_teacher_outputs(self, batch):
        """Teacher 모델의 출력 얻기"""
        signal, signal_len, transcript, transcript_len = batch
        
        # Preprocessor
        processed_signal, processed_signal_len = self.teacher_model.preprocessor(
            input_signal=signal, length=signal_len
        )
        
        # Spec augment (평가 모드이므로 적용 안됨)
        if self.teacher_model.spec_augmentation is not None and self.teacher_model.spec_augmentation.training:
            processed_signal = self.teacher_model.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_len
            )
        
        # Encoder
        encoded, encoded_len = self.teacher_model.encoder(
            audio_signal=processed_signal, length=processed_signal_len
        )
        
        # CTC Decoder (logits)
        ctc_logits = None
        if hasattr(self.teacher_model, 'ctc_decoder'):
            ctc_logits = self.teacher_model.ctc_decoder(encoder_output=encoded)
        
        return {
            'encoded': encoded,
            'encoded_len': encoded_len,
            'ctc_logits': ctc_logits,
        }
    
    def _get_student_outputs(self, batch):
        """Student 모델의 출력 얻기"""
        signal, signal_len, transcript, transcript_len = batch
        
        # Preprocessor
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=signal, length=signal_len
        )
        
        # Spec augment
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_len
            )
        
        # Encoder
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_len
        )
        
        # CTC Decoder (logits)
        ctc_logits = None
        if hasattr(self, 'ctc_decoder'):
            ctc_logits = self.ctc_decoder(encoder_output=encoded)
        
        return {
            'encoded': encoded,
            'encoded_len': encoded_len,
            'ctc_logits': ctc_logits,
        }
    
    def _compute_distillation_loss(self, student_outputs, teacher_outputs):
        """
        Distillation loss 계산
        
        1. Encoder output distillation (MSE)
        2. CTC logits distillation (KL divergence)
        """
        total_distill_loss = 0.0
        num_losses = 0
        
        # 1. Encoder output distillation (feature matching)
        if self.use_encoder_distill:
            student_encoded = student_outputs['encoded']
            teacher_encoded = teacher_outputs['encoded']
            
            # MSE loss between encoder outputs
            encoder_distill_loss = F.mse_loss(
                student_encoded, 
                teacher_encoded,
                reduction='mean'
            )
            
            total_distill_loss += encoder_distill_loss
            num_losses += 1
            
            self.log('distill_loss_encoder', encoder_distill_loss, prog_bar=False, sync_dist=True)
        
        # 2. CTC Logits distillation (KL divergence)
        if self.use_logits_distill and student_outputs['ctc_logits'] is not None:
            student_logits = student_outputs['ctc_logits']
            teacher_logits = teacher_outputs['ctc_logits']
            
            # Temperature scaling
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # KL divergence
            logits_distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            total_distill_loss += logits_distill_loss
            num_losses += 1
            
            self.log('distill_loss_logits', logits_distill_loss, prog_bar=False, sync_dist=True)
        
        # Average
        if num_losses > 0:
            total_distill_loss = total_distill_loss / num_losses
        
        return total_distill_loss


@hydra_runner(
    config_path="../conf/fastconformer/hybrid_transducer_ctc/", 
    config_name="fastconformer_hybrid_transducer_ctc_bpe_small"
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    
    # Distillation 설정 추가
    if 'distillation' not in cfg:
        with open_dict(cfg):
            cfg.distillation = {
                'alpha': 0.5,
                'temperature': 2.0,
                'use_encoder_distill': True,
                'use_logits_distill': True,
            }
    
    # Teacher 모델 경로 확인
    teacher_model_path = cfg.get('teacher_model_path', None)
    if teacher_model_path is None:
        logging.warning("⚠️  teacher_model_path not specified! Training without distillation.")
        logging.warning("   To use distillation, add: teacher_model_path=<path/to/teacher.nemo>")
    else:
        logging.info(f"📚 Knowledge Distillation enabled")
        logging.info(f"   Teacher model: {teacher_model_path}")
        logging.info(f"   Alpha (student weight): {cfg.distillation.alpha}")
        logging.info(f"   Temperature: {cfg.distillation.temperature}")
    
    # Trainer 생성
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Student 모델 생성 (Distillation 지원)
    asr_model = DistillationHybridRNNTCTCBPEModel(
        cfg=cfg.model, 
        teacher_model_path=teacher_model_path,
        trainer=trainer
    )
    
    # Initialize from pretrained checkpoint if provided
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # 학습 시작
    logging.info("🚀 Starting distillation training...")
    trainer.fit(asr_model)
    
    # 테스트
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)
    
    logging.info("✓ Training complete!")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
