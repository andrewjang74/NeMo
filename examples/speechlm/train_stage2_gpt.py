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

import nemo_run as run
from nemo.collections import llm
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to directory containing tokenizer.model")
    parser.add_argument("--tokenizer_type", type=str, default="bpe", help="Type of tokenizer (bpe or wpe)")
    args = parser.parse_args()

    # 1. Define the Small GPT Config (~60MB)
    # 12 layers, 512 hidden, 8 heads -> Approx 30M params (FP16) or 60MB weights.
    gpt_config = llm.GPTConfig(
        num_layers=12,
        hidden_size=512,
        num_attention_heads=8,
        max_position_embeddings=2048,
        ffn_hidden_size=2048,
        vocab_size=32000, # Ensure this matches your tokenizer vocab size
        activation_func='gelu',
        bias=True,
        normalization='layernorm',
        layernorm_epsilon=1e-5,
    )

    # 2. Define the Tokenizer
    # Loading tokenizer from the specified directory (same as Stage 1)
    if args.tokenizer_type == "bpe":
        tokenizer_model_path = f"{args.tokenizer_dir}/tokenizer.model"
        # Assuming SentencePiece tokenizer for BPE
        tokenizer = llm.Tokenizer(tokenizer_model_path) 
    else:
        # Handle other types if necessary, or default to AutoTokenizer for HF
        tokenizer = llm.AutoTokenizer(args.tokenizer_dir)

    # 3. Define the Model
    model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

    # 4. Define Data
    # Replace MockDataModule with your actual PretrainingDataModule
    # data = llm.PretrainingDataModule(paths=[...], seq_length=2048, ...)
    data = llm.MockDataModule(seq_length=2048, global_batch_size=8, micro_batch_size=4)

    # 5. Define Trainer and Recipe
    recipe = llm.pretrain(
        model=model,
        data=data,
        trainer=run.Config(
            "nemo.lightning.Trainer",
            devices=1,
            accelerator="gpu",
            max_steps=10000,
            val_check_interval=1000,
            log_every_n_steps=10,
            limit_val_batches=5,
        ),
        optim=run.Config(
            "nemo.lightning.MegatronOptimizerModule",
            config=run.Config(
                "megatron.core.optimizer.OptimizerConfig", 
                lr=6e-4, 
                optimizer="adam",
                weight_decay=0.1,
                bf16=True
            ),
            lr_scheduler=run.Config(
                "nemo.lightning.pytorch.optim.CosineAnnealingScheduler",
                warmup_steps=100,
                min_lr=6e-5,
                constant_steps=0,
            )
        ),
        log=run.Config(
            "nemo.lightning.NeMoLogger", 
            name="stage2_gpt_small",
            tensorboard=run.Config("lightning.pytorch.loggers.TensorBoardLogger", save_dir="nemo_experiments"),
            ckpt=run.Config(
                "nemo.lightning.pytorch.callbacks.ModelCheckpoint",
                save_last=True,
                save_top_k=1,
                filename='{epoch}-{step}',
                monitor='val_loss',
                mode='min',
                always_save_context=True,
            )
        ),
    )

    # 6. Run Training
    print("Starting Stage 2: Small GPT Pretraining...")
    run.run(recipe)

if __name__ == "__main__":
    main()
