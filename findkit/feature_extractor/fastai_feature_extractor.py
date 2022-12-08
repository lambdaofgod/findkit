from dataclasses import dataclass, field
from typing import Callable, List, Union

import numpy as np
import toolz
import torch
import torch.utils.data
import tqdm
from fastai.text import all as text
from findkit.util import masked_mean_pool

from .feature_extractor import FeatureExtractor

SentenceEncoderInput = Union[str, List[str]]

PoolingFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class FastAITextFeatureExtractor(FeatureExtractor[SentenceEncoderInput]):

    max_length: int
    fastai_learner: text.LMLearner
    tokenizer: text.Tokenizer
    numericalizer: text.Numericalize
    pad_token_idx: int = field(default=2)
    dataloader_num_workers: int = field(default=0)
    batch_size: int = field(default=256)
    pooling_fns: List[PoolingFn] = field(default_factory=lambda: [masked_mean_pool])
    device: str = field(default="cuda")

    @staticmethod
    def build_from_learner(
        fastai_learner: text.LMLearner,
        max_length: int,
        pad_token_idx: int = 2,
        batch_size: int = 256,
        pooling_fns: List[PoolingFn] = [masked_mean_pool],
        dataloader_num_workers: int = 0,
        device: str = "cuda",
    ):
        tokenizer = fastai_learner.dls.tokenizer
        numericalizer = fastai_learner.dls.tfms[-1]
        return FastAITextFeatureExtractor(
            max_length,
            fastai_learner,
            tokenizer,
            numericalizer,
            pad_token_idx,
            dataloader_num_workers,
            batch_size,
            pooling_fns,
            device,
        )

    def extract_features(
        self,
        data: SentenceEncoderInput,
        show_progress_bar=True,
    ):
        dataloader = self._get_dataloader(data)
        embs = []

        model = self._get_model()
        with torch.no_grad():
            if show_progress_bar:
                batch_iterator = tqdm.auto.tqdm(dataloader)
            else:
                batch_iterator = dataloader
            for b, b_mask in batch_iterator:
                b_embs = self._extract_batch_features(model, b, b_mask)
                embs.append(b_embs)
        return np.row_stack(embs)

    def _extract_batch_features(self, model, data_batch, mask_batch):
        raw_embs = model(data_batch.to(self.device))
        mask_batch = mask_batch.to(self.device)
        masked_embs = raw_embs * mask_batch.unsqueeze(-1)
        return np.row_stack(
            [
                pool(masked_embs, mask_batch, axis=1).cpu().numpy()
                for pool in self.pooling_fns
            ]
        ).astype(np.float16)

    def _get_model(self):
        model = self.fastai_learner.model[0].to(self.device)
        model.reset()
        model.eval()
        return model

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

    def save(self, path):
        pass

    def load(self, path):
        pass


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
