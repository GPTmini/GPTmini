"""Shared hyperparameters and file paths for the pretrain -> SFT -> DPO pipeline."""

# data / checkpoint paths
DATA_FILE = "../data/tinyshakespeare.txt"
SFT_SAMPLES = "../data/sft-samples.pkl"
DPO_SAMPLES = "../data/dpo-samples.pkl"

GPT_MODEL = "../data/tinyshakespeare-gpt.npz"
SFT_MODEL = "../data/tinyshakespeare-sft.npz"
DPO_MODEL = "../data/tinyshakespeare-dpo.npz"

# model architecture
BATCH_SIZE = 4
CONTEXT_SIZE = 128
EMBEDDING_SIZE = 320
HEADS = 4
BLOCKS = 6
DROPOUT = 0.1

# pretraining LR schedule
GPT_MAX_LR = 0.0005
GPT_MIN_LR = 0.0001
GPT_WARMUP_STEPS = 300

# SFT LR schedule (smaller LR: fine-tuning from a pretrained checkpoint)
SFT_MAX_LR = 0.00005
SFT_MIN_LR = 0.00001
SFT_WARMUP_STEPS = 50

# DPO LR schedule
DPO_MAX_LR = 0.000005
DPO_MIN_LR = 0.000001
DPO_WARMUP_STEPS = 50

EPOCHS = 10
PROMPT = "ROMEO:"  # sample prompt used to preview generations during training
