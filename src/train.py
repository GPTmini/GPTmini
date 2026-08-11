"""Entry point for the full pipeline: GPT pretraining -> SFT -> DPO,
each stage building on the checkpoint saved by the previous one."""

from pathlib import Path

import numpy as np
import requests

from src.config import *
from src.core.lr_scheduler import WarmupCosineScheduler
from src.core.optimizer import AdamWOptimizer
from src.gpt.dpo_dataset import DPODataset
from src.gpt.dpo_loss import DPOLoss
from src.gpt.dpo_model import DPOModel
from src.gpt.gpt import GPT
from src.gpt.gpt_dataset import GPTDataset
from src.gpt.gpt_loss import GPTLoss
from src.gpt.gpt_model import GPTModel
from src.gpt.sft_dataset import SFTDataset
from src.gpt.sft_loss import SFTLoss
from src.gpt.sft_model import SFTModel
from src.sample import dpo_sample, sft_sample


def gpt_train():
    """Stage 1: pretrain a GPT from scratch on raw text (next-token prediction)."""
    file = Path(DATA_FILE)
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()
        file.write_text(response.text)

    dataset = GPTDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    print(f"Dataset: vocab_size={dataset.vocab_size} train_samples={len(dataset)}")

    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, DROPOUT)
    loss_fn = GPTLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=GPT_MAX_LR)
    model = GPTModel(GPT_MODEL, layer, loss_fn, optimizer)  # resumes from GPT_MODEL if it exists
    print(f"Model: parameters={sum(p.data.size for p in layer.parameters)}, resumed at step {model.steps}")

    scheduler = WarmupCosineScheduler(GPT_MAX_LR, model.steps + EPOCHS * len(dataset), GPT_WARMUP_STEPS, GPT_MIN_LR)
    model.train(dataset, EPOCHS, scheduler=scheduler, prompt=PROMPT)


def sft_train():
    """Stage 2: fine-tune the pretrained GPT on (prompt, response) pairs."""
    dataset = GPTDataset(DATA_FILE, 1, CONTEXT_SIZE)
    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, DROPOUT)
    loss_fn = SFTLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=SFT_MAX_LR)
    model = SFTModel(GPT_MODEL, layer, loss_fn, optimizer)  # start from the pretrained checkpoint
    model.filename = SFT_MODEL  # but save to a different file
    model.steps = 0  # reset step count; this is a fresh training stage, not a resume

    sft_dataset = SFTDataset(SFT_SAMPLES, 1, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(SFT_MAX_LR, len(sft_dataset), SFT_WARMUP_STEPS, SFT_MIN_LR)
    model.train(sft_dataset, 1, scheduler=scheduler)

    text = model.test(dataset, prompt=PROMPT)
    print(text)


def dpo_train():
    """Stage 3: align the SFT model with preference pairs via DPO."""
    dataset = GPTDataset(DATA_FILE, 1, CONTEXT_SIZE)
    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, DROPOUT)
    reference = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, DROPOUT)
    loss_fn = DPOLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=DPO_MAX_LR)
    model = DPOModel(SFT_MODEL, layer, loss_fn, optimizer, reference)  # policy + reference both start from SFT
    model.filename = DPO_MODEL
    model.steps = 0  # reset step count; this is a fresh training stage, not a resume

    dpo_dataset = DPODataset(DPO_SAMPLES, 1, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(DPO_MAX_LR, len(dpo_dataset), DPO_WARMUP_STEPS, DPO_MIN_LR)
    model.train(dpo_dataset, 1, scheduler=scheduler)

    text = model.test(dataset, prompt=PROMPT)
    print(text)


if __name__ == "__main__":
    np.random.seed(42)

    gpt_train()

    sft_sample()
    sft_train()

    dpo_sample()
    dpo_train()
