from dataclasses import dataclass, field
from typing import Callable, List, Union

import numpy as np
import toolz
import torch
import torch.utils.data
import tqdm
from fastai.text import all as text

from .feature_extractor import FeatureExtractor

SentenceEncoderInput = Union[str, List[str]]

PoolingFn = Callable[[torch.Tensor], torch.Tensor]


def masked_mean_pool(inputs: torch.Tensor, mask: torch.Tensor, axis: int = 1):
    if len(mask.shape) != len(inputs.shape):
        mask = mask.unsqueeze(-1)
    mask = mask.to(inputs.device)
    return (inputs * mask).sum(axis=1) / mask.sum(axis=1)


def masked_max_pool(inputs: torch.Tensor, mask: torch.Tensor, axis: int = 1):
    if len(mask.shape) != len(inputs.shape):
        mask = mask.unsqueeze(-1)
    mask = mask.to(inputs.device)
    return (inputs * mask).max(axis=1)


@dataclass
class FastaiTextFeatureExtractor(FeatureExtractor[SentenceEncoderInput]):

    max_length: int
    fastai_learner: text.LMLearner
    tokenizer: text.Tokenizer
    numericalizer: text.Numericalize
    pad_token_idx: int = field(default=2)
    dataloader_num_workers: int = field(default=0)
    batch_size: int = field(default=256)
    pooling_fns: List[PoolingFn] = field(default_factory=lambda: [masked_mean_pool])
    device: str = field(default="cuda")

    def extract_features(
        self,
        data: SentenceEncoderInput,
    ):
        dataloader = self._get_dataloader(data)
        embs = []
        with torch.no_grad():
            for b, b_mask in tqdm.auto.tqdm(dataloader):
                b_embs = self._extract_batch_features(b, b_mask)
                embs.append(b_embs)
        return np.row_stack(embs)

    def _extract_batch_features(self, data_batch, mask_batch):
        model = self.fastai_learner.model[0].to(self.device)
        model.reset()
        raw_embs = model(data_batch.to(self.device))
        mask_batch = mask_batch.to(self.device)
        masked_embs = raw_embs * mask_batch.unsqueeze(-1)
        return np.row_stack(
            [
                pool(masked_embs, mask_batch, axis=1).cpu().numpy()
                for pool in self.pooling_fns
            ]
        )

    def _get_dataset(self, data):
        return FastaiSequenceDataset(
            data, tokenizer=self.tokenizer, numericalizer=self.numericalizer
        )

    def _get_dataloader(self, data):
        collate_fn = toolz.partial(
            get_padded_batch_with_mask, max_length=self.max_length, pad_token_idx=2
        )
        return torch.utils.data.DataLoader(
            self._get_dataset(data),
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
        )


class FastaiSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts,
        tokenizer: text.Tokenizer,
        numericalizer: text.Numericalize,
        metadata=None,
    ):
        self.texts = texts
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.numericalizer = numericalizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        tokens = self.tokenizer(t)
        return self.numericalizer(tokens)


def get_padded_batch_with_mask(inputs, max_length=None, pad_token_idx=2):
    lengths = [len(n) for n in inputs]
    if max_length is None:
        max_length = max(lengths)
    mask = torch.zeros((len(inputs), max_length))
    inputs_tensors = (
        torch.ones((len(inputs), max_length), dtype=torch.int64) * pad_token_idx
    )
    for i, l in enumerate(lengths):
        input = inputs[i]
        step_max_length = min(len(input), max_length)
        inputs_tensors[i][:step_max_length] = input[:step_max_length]
        mask[i][:l] = 1
    return inputs_tensors, mask
