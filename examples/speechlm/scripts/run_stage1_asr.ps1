# Stage 1: Train FastConformer ASR
# Ensure you have your manifest files ready.
# Replace MANIFEST_PATH with your actual paths.

$TRAIN_MANIFEST = "data/train.json"
$VAL_MANIFEST = "data/val.json"

# Using the config file you are currently viewing
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py `
    --config-path="examples/asr/conf/fastconformer/hybrid_transducer_ctc" `
    --config-name="fastconformer_hybrid_transducer_ctc_bpe_l12_half" `
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST `
    model.validation_ds.manifest_filepath=$VAL_MANIFEST `
    exp_manager.name="stage1_asr" `
    exp_manager.exp_dir="nemo_experiments/stage1_asr"
