from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ..index.index import Index

try:
    import pynndescent

    PyNNDescentIndexImpl = pynndescent.NNDescent
except ModuleNotFoundError:
    PyNNDescentIndexImpl = "pynndescent not found"


@dataclass(frozen=True)
class NNDescentIndex(Index):

    _index: pynndescent.NNDescent
    _metadata: pd.DataFrame
    _dimensionality: int

    def metadata(self):
        return self._metadata

    def dimensionality(self):
        return self._dimensionality

    @staticmethod
    def build(data, metadata=None, **kwargs):
        metadata = Index._get_valid_metadata(data, metadata)
        nnd_index = pynndescent.NNDescent(data, **kwargs)
        return NNDescentIndex(nnd_index, metadata, data.shape[1])

    def find_similar_raw(
        self, query_object: np.ndarray, n_returned: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._index.rng_state = np.array([42, 42, 42], dtype=np.int64)
        self.validate_input_data(query_object)
        indices, dists = self._index.query(query_object.reshape(1, -1), k=n_returned)
        return indices.reshape(-1), dists.reshape(-1)

    def get_dimensionality(self):
        return self.dimensionality

    def _get_config(self) -> dict:
        """
        get config for saving and loading from disk
        """
        return {"_dimensionality": self._dimensionality}

    def _save_index(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._index, f)

    @classmethod
    def _load_from_disk(self, path: str, config: dict, metadata: pd.DataFrame):
        with open(path, "rb") as f:
            _index = pickle.load(f)
        return NNDescentIndex(_index, metadata, **config)
