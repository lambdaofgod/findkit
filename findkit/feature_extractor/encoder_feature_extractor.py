from typing import List, Protocol, Union

from dataclasses import dataclass
import numpy as np

from .feature_extractor import FeatureExtractor


SentenceEncoderInput = Union[str, List[str]]


class SentenceEncoder(Protocol):
    """
    protocol for types that have defined `encode` method
    like SentenceTransformer class
    """

    def encode(self, sentences: SentenceEncoderInput) -> np.ndarray:
        pass


@dataclass(frozen=True)
class SentenceEncoderFeatureExtractor(FeatureExtractor[SentenceEncoderInput]):

    encoder: SentenceEncoder

    def extract_features(self, data: SentenceEncoderInput, **kwargs):
        return self.encoder.encode(data, **kwargs)
