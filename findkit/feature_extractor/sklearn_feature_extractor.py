from typing import List, Protocol, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse.base import spmatrix

from .feature_extractor import FeatureExtractor

MatrixInput = Union[np.ndarray, pd.DataFrame, List[str], spmatrix]


class SklearnTransformer(Protocol):
    """
    protocol for scikit-learn classes that can be used to:
    - extract features from texts
    - reduce dimensionality of data
    """

    def transform(self, X: MatrixInput) -> np.ndarray:
        pass


@dataclass(frozen=True)
class SklearnFeatureExtractor(FeatureExtractor[MatrixInput]):
    """
    Wrapper for scikit-learn style transformers

    Parameters
    ----------
    sklearn_transformer: object with `transform` or with `fit_transform` method

    note that we need both of these as some transformers are nonparametri
    (for example tSNE or UMAP)
    """

    transformer: SklearnTransformer

    def extract_features(self, data: MatrixInput, **kwargs):
        if type(data) is str:
            return self.transformer.transform([data])
        if len(data.shape) == 1:
            return self.transformer.transform([data])
        else:
            return self.transformer.transform(data)

    def fit_extract_features(self, data: MatrixInput, **kwargs):
        return self.transformer.fit_transform(data)
