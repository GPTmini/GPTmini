import pickle
import re
from itertools import pairwise

import numpy as np

from src.core import AdamWOptimizer, WarmupCosineScheduler
from src.gpt import CharDataset, GPT
from src.gpt.gpt import DATA_FILE, CONTEXT_SIZE, BATCH_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, MODEL_FILE
from src.sft.sft_dataset import SFTDataset
from src.sft.sft_loss import SFTLoss
from src.sft.sft_model import SFTModel

np.random.seed(42)

SFT_SAMPLES = "../sft-samples.pkl"
SFT_MODEL = "../tinyshakespeare-sft.npz"

SFT_MAX_LR = 0.00005
SFT_MIN_LR = 0.00001
SFT_WARMUP_STEPS = 50


def extract_turns(max_turn_chars=100):
    with open(DATA_FILE, encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"\n\s*\n", text.strip())
    turns = []

    for block in blocks:
        lines = block.strip("\n").split("\n")
        if not lines:
            continue

        m = re.compile(r"^([A-Z][A-Za-z' ]{0,30}):\s*$").match(lines[0].strip())
        if not m:
            continue

        speaker = m.group(1).strip()
        content = " ".join(line.strip() for line in lines[1:] if line.strip())
        if not content:
            continue

        turns.append((speaker, content[:max_turn_chars]))

    return turns


def sft_sample():
    dataset = CharDataset(DATA_FILE, 1, CONTEXT_SIZE)

    pairs = list(pairwise(extract_turns()))
    np.random.shuffle(pairs)
    pairs = pairs[:1024]

    samples = []
    for (speaker_a, content_a), (speaker_b, content_b) in pairs:
        prompt = f"{speaker_a}:\n{content_a}\n"
        response = f"{speaker_b}:\n{content_b}\n"
        samples.append((dataset.encode(prompt), dataset.encode(response)))

    with open(SFT_SAMPLES, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} SFT samples to {SFT_SAMPLES}")

    sample = samples[0]
    print(f"{dataset.decode(sample[0])}\n{dataset.decode(sample[1])}")


def sft_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = SFTLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=SFT_MAX_LR)
    model = SFTModel(layer, loss_fn, optimizer)
    model.load(MODEL_FILE)

    sft_dataset = SFTDataset(SFT_SAMPLES, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(SFT_MAX_LR, len(sft_dataset), SFT_WARMUP_STEPS, SFT_MIN_LR)
    model.train(sft_dataset, 1, scheduler, SFT_MODEL)
    return model, dataset


def sft_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    sft_sample()
    model, dataset = sft_train()
    sft_generate(model, dataset)
