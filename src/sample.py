"""Builds SFT and DPO training samples from the raw dialogue text.

SFT samples: consecutive (speaker A, speaker B) turns become (prompt, response).
DPO samples: same prompts, but "chosen" is the real next turn and "rejected"
is sampled from the (not-yet-aligned) GPT model itself.
"""

import pickle
import re
from itertools import pairwise

import numpy as np

from src.config import *
from src.core.tensor import Tensor
from src.gpt.gpt import GPT
from src.gpt.gpt_dataset import GPTDataset
from src.gpt.gpt_loss import GPTLoss
from src.gpt.gpt_model import GPTModel
from src.gpt.optimizer import AdamWOptimizer


def sft_sample():
    dataset = GPTDataset(DATA_FILE, 1, CONTEXT_SIZE)

    examples = _build_sft_examples(dataset)

    with open(SFT_SAMPLES, "wb") as f:
        pickle.dump(examples, f)
    print(f"saved {len(examples)} SFT samples to {SFT_SAMPLES}")


def dpo_sample():
    dataset = GPTDataset(DATA_FILE, 1, CONTEXT_SIZE)

    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS, 0.0)
    loss_fn = GPTLoss()
    optimizer = AdamWOptimizer(layer.parameters(), lr=GPT_MAX_LR)
    GPTModel(GPT_MODEL, layer, loss_fn, optimizer)  # loads pretrained weights into `layer`
    layer.eval()

    examples = _build_dpo_examples(dataset, layer)

    with open(DPO_SAMPLES, "wb") as f:
        pickle.dump(examples, f)
    print(f"saved {len(examples)} DPO samples to {DPO_SAMPLES}")


def _build_sft_examples(dataset, max_turn_chars=100, max_pairs=1024):
    """Each consecutive pair of turns becomes one (prompt, response) example."""
    turns = _extract_turns(max_turn_chars, max_pairs)

    examples = []
    for (speaker_a, content_a), (speaker_b, content_b) in pairwise(turns):
        prompt = f"{speaker_a}:\n{content_a}\n"
        response = f"{speaker_b}:\n{content_b}\n"
        examples.append((dataset.encode(prompt), dataset.encode(response)))

    return examples


def _build_dpo_examples(dataset, layer, max_turn_chars=100, batches=64, max_pairs=1024):
    """Chosen = the real next turn; rejected = sampled from `layer` given the
    same prompt. Pairs where sampling produced an empty or identical
    continuation are dropped."""
    turns = _extract_turns(max_turn_chars, max_pairs)
    pairs = list(pairwise(turns))

    examples = []
    for start in range(0, len(pairs), batches):
        chunk = pairs[start: start + batches]

        prompt_tokens = [dataset.encode(f"{a}:\n{t}\n") for (a, t), (_, _) in chunk]
        chosen_tokens = [dataset.encode(f"{b}:\n{t}\n") for (_, _), (b, t) in chunk]

        lens = [len(c) for c in chosen_tokens]
        rejected_tokens = _generate_rejected_batch(layer, prompt_tokens, lens)

        for p, c, r in zip(prompt_tokens, chosen_tokens, rejected_tokens):
            if len(r) == 0 or r == c:
                continue
            examples.append((p, c, r))

        print(f"generated {len(examples)}/{len(pairs)} DPO samples")

    return examples


def _extract_turns(max_turn_chars, max_pairs=1024):
    """Parse "SPEAKER:\\ntext..." blocks out of the raw play text, one per turn,
    truncating overly long turns at a sentence boundary where possible."""
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

        if len(content) > max_turn_chars:
            cut = content.rfind(". ", 0, max_turn_chars)
            content = content[: cut + 1] if cut != -1 else content[:max_turn_chars]
        turns.append((speaker, content))

    if max_pairs is not None:
        np.random.shuffle(turns)
        turns = turns[:max_pairs]
    return turns


def _generate_rejected_batch(layer, prompts, lens, pad_token=0, temperature=0.8, top_k=20):
    """Autoregressively sample a continuation for each prompt in the batch
    (same top-k + temperature sampling as GPTModel.test), used as the
    "rejected" response for DPO. `pad_token` here is only a left-align filler
    so prompts of different lengths can be stacked into one batch for speed;
    it's independent of SFTDataset (which doesn't pad) and the model being
    sampled from (the plain pretrained GPT) was never trained on it either,
    so its exact value doesn't matter much for a quick negative sample."""
    sequences = [list(p) for p in prompts]

    with Tensor.no_grad():
        for _ in range(max(lens)):
            windows = [(([pad_token] * CONTEXT_SIZE) + seq)[-CONTEXT_SIZE:] for seq in sequences]
            batch = np.array(windows, dtype=np.int64)
            logits = layer(Tensor(batch))
            next_logits = logits.data[:, -1, :].astype(np.float64) / temperature

            k = min(top_k, next_logits.shape[-1])
            for row in range(len(prompts)):
                row_logits = next_logits[row]
                threshold = np.partition(row_logits, -k)[-k]
                row_logits = np.where(row_logits < threshold, -np.inf, row_logits)
                probs = np.exp(row_logits - np.max(row_logits))
                probs /= np.sum(probs)
                sequences[row].append(int(np.random.choice(len(probs), p=probs)))

    return [seq[len(prompts[i]): len(prompts[i]) + lens[i]] for i, seq in enumerate(sequences)]
