"""DPO dataset: reuses SFTDataset's windowing, but each example is
(prompt, chosen, rejected) and gets packed into a chosen and a rejected
sequence side by side."""

from src.core.tensor import Tensor
from src.gpt.sft_dataset import SFTDataset


class DPODataset(SFTDataset):

    def _pack(self, examples):
        cxs, cys, cmasks, rxs, rys, rmasks = [], [], [], [], [], []
        for prompt, chosen, rejected in examples:
            x, y, m = self._build_example(prompt, chosen)
            cxs.append(x)
            cys.append(y)
            cmasks.append(m)
            x, y, m = self._build_example(prompt, rejected)
            rxs.append(x)
            rys.append(y)
            rmasks.append(m)
        return cxs, cys, cmasks, rxs, rys, rmasks

    def __getitem__(self, index):
        s = self._slice(index)
        cx, cy, cmask, rx, ry, rmask = self.data
        return (Tensor(cx[s]), Tensor(cy[s]), Tensor(cmask[s]),
                Tensor(rx[s]), Tensor(ry[s]), Tensor(rmask[s]))
