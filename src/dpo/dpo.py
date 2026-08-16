import pickle
from itertools import pairwise

import numpy as np

from src.core import WarmupCosineScheduler, AdamWOptimizer, Tensor
from src.dpo.dpo_dataset import DPODataset
from src.dpo.dpo_loss import DPOLoss
from src.dpo.dpo_model import DPOModel
from src.gpt import CharDataset, GPT, GPTModel
from src.gpt.gpt import CONTEXT_SIZE, MODEL_FILE, EMBEDDING_SIZE, HEADS, BLOCKS, BATCH_SIZE, DATA_FILE
from src.sft.sft import extract_turns

np.random.seed(42)

DPO_SAMPLES = "../dpo-samples.pkl"
DPO_MODEL = "../tinyshakespeare-dpo.npz"

DPO_MAX_LR = 0.000005
DPO_MIN_LR = 0.000001
DPO_WARMUP_STEPS = 50


def generate_rejected_batch(layer, prompts, lens, pad_token=0, temperature=0.8, top_k=20):
    sequences = [list(p) for p in prompts]

    with Tensor.no_grad():
        for _ in range(max(lens)):
            windows = [(([pad_token] * CONTEXT_SIZE) + seq)[-CONTEXT_SIZE:] for seq in sequences]
            batch = np.array(windows, dtype=np.int64)
            logits = layer(Tensor(batch))
            next_logits = logits.data[:, -1, :] / temperature

            k = min(top_k, next_logits.shape[-1])
            for row in range(len(prompts)):
                row_logits = next_logits[row]
                threshold = np.partition(row_logits, -k)[-k]
                row_logits = np.where(row_logits < threshold, -np.inf, row_logits)
                probs = np.exp(row_logits - np.max(row_logits))
                probs /= np.sum(probs)
                sequences[row].append(int(np.random.choice(len(probs), p=probs)))

    return [seq[len(prompts[i]): len(prompts[i]) + lens[i]] for i, seq in enumerate(sequences)]


def dpo_sample():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)

    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    model = GPTModel(layer, None, None)
    model.load(MODEL_FILE)
    layer.eval()

    pairs = list(pairwise(extract_turns()))
    np.random.shuffle(pairs)
    pairs = pairs[:1024]

    samples = []
    for start in range(0, len(pairs), 64):
        chunk = pairs[start: start + 64]

        prompt_tokens = [dataset.encode(f"{a}:\n{t}\n") for (a, t), (_, _) in chunk]
        chosen_tokens = [dataset.encode(f"{b}:\n{t}\n") for (_, _), (b, t) in chunk]

        lens = [len(c) for c in chosen_tokens]
        rejected_tokens = generate_rejected_batch(layer, prompt_tokens, lens)

        for p, c, r in zip(prompt_tokens, chosen_tokens, rejected_tokens):
            if len(r) == 0 or r == c:
                continue
            samples.append((p, c, r))

        print(f"generated {len(samples)}/{len(pairs)} DPO samples")

    with open(DPO_SAMPLES, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} DPO samples to {DPO_SAMPLES}")

    sample = samples[0]
    print(f"{dataset.decode(sample[0])}\n{dataset.decode(sample[1])}\n{dataset.decode(sample[2])}")


def dpo_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    reference = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = DPOLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=DPO_MAX_LR)
    model = DPOModel(layer, loss_fn, optimizer, reference)
    model.load(MODEL_FILE)
    model.load_reference(MODEL_FILE)

    dpo_dataset = DPODataset(DPO_SAMPLES, CONTEXT_SIZE)
    scheduler = WarmupCosineScheduler(DPO_MAX_LR, len(dpo_dataset), DPO_WARMUP_STEPS, DPO_MIN_LR)
    model.train(dpo_dataset, 1, scheduler, DPO_MODEL)
    return model, dataset


def dpo_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    dpo_sample()
    model, dataset = dpo_train()
    dpo_generate(model, dataset)
