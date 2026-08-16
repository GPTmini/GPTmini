from pathlib import Path

import numpy as np
import requests

from src.core import AdamWOptimizer, WarmupCosineScheduler
from src.gpt import CharDataset, GPT, GPTModel, GPTLoss

np.random.seed(42)

DATA_FILE = "../tinyshakespeare.txt"
MODEL_FILE = "../tinyshakespeare-gpt.npz"

BATCH_SIZE = 4
CONTEXT_SIZE = 128
EMBEDDING_SIZE = 320
HEADS = 4
BLOCKS = 6
EPOCHS = 10

GPT_MAX_LR = 0.0005
GPT_MIN_LR = 0.0001
GPT_WARMUP_STEPS = 300


def download_file():
    file = Path(DATA_FILE)
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()
        file.write_text(response.text)


def gpt_train():
    dataset = CharDataset(DATA_FILE, BATCH_SIZE, CONTEXT_SIZE)
    layer = GPT(dataset.vocab_size, CONTEXT_SIZE, EMBEDDING_SIZE, HEADS, BLOCKS)
    loss_fn = GPTLoss()
    optimizer = AdamWOptimizer(layer.parameters, lr=GPT_MAX_LR)
    model = GPTModel(layer, loss_fn, optimizer)

    scheduler = WarmupCosineScheduler(GPT_MAX_LR, EPOCHS * len(dataset), GPT_WARMUP_STEPS, GPT_MIN_LR)
    model.train(dataset, EPOCHS, scheduler, MODEL_FILE)
    return model, dataset


def gpt_test(model, dataset):
    prediction, loss = model.test(dataset)
    print(f'prediction: {len(prediction)} steps, each {prediction[0].shape}')
    print(f'loss: {loss}')


def gpt_generate(model, dataset):
    print(model.generate(dataset, prompt="ROMEO:"))


if __name__ == "__main__":
    download_file()
    model, dataset = gpt_train()
    gpt_test(model, dataset)
    gpt_generate(model, dataset)
