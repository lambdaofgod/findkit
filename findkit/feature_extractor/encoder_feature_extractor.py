from typing import List, Protocol, Union

from dataclasses import dataclass
import numpy as np

from .feature_extractor import FeatureExtractor


class SentenceEncoder(Protocol):
    """
    protocol for types that have defined `encode` method
    like SentenceTransformer class
    """

    def encode(self, sentences: Union[str, List[str]]) -> np.ndarray:
        pass


@dataclass(frozen=True)
class SentenceEncoderFeatureExtractor(FeatureExtractor):

    encoder: SentenceEncoder

    def extract_features(self, data: Union[str, List[str]], **kwargs):
        return self.encoder.encode(data, **kwargs)
