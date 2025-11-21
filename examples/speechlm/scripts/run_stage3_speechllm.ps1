# Stage 3: Train SpeechLLM (Combined)
# Point to the checkpoints from Stage 1 and Stage 2.

# Path to the .nemo file from Stage 1 (ASR)
# Update this path to your actual Stage 1 output
$ASR_CHECKPOINT = "nemo_experiments/stage1_asr/checkpoints/stage1_asr.nemo"

# Path to the checkpoint directory or .nemo file from Stage 2 (LLM)
# Update this path to your actual Stage 2 output
$LLM_CHECKPOINT = "nemo_experiments/stage2_gpt_small/checkpoints/last.ckpt" 

# Manifests for Speech-to-Text training (Audio + Text)
$TRAIN_MANIFEST = "data/speech_text_train.json"
$VAL_MANIFEST = "data/speech_text_val.json"

python examples/speechlm/speech_to_text_llm_train.py `
    --config-path="examples/speechlm/conf/custom" `
    --config-name="stage3_speechllm" `
    model.speech_encoder.pretrained_model=$ASR_CHECKPOINT `
    model.llm.pretrained_model=$LLM_CHECKPOINT `
    data.train_ds.manifest_filepath=$TRAIN_MANIFEST `
    data.validation_ds.manifest_filepath=$VAL_MANIFEST `
    name="stage3_speechllm"
