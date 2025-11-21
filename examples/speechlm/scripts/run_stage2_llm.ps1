# Stage 2: Train Small GPT LLM
# This script uses MockData by default. 
# Please edit examples/speechlm/train_stage2_gpt.py to use your real text dataset.

# Path to the directory containing tokenizer.model (same as Stage 1)
$TOKENIZER_DIR = "tokenizers/bpe_32k" 

python examples/speechlm/train_stage2_gpt.py --tokenizer_dir $TOKENIZER_DIR --tokenizer_type "bpe"
