from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pandas as pd

from ..index.index import Index

try:
    import nmslib

    NMSLIBFloatIndex = nmslib.dist.FloatIndex
except ModuleNotFoundError:
    NMSLIBFloatIndex = "NMSLib not found"


@dataclass(frozen=True)
class NMSLIBIndex(Index):
    """
    Index using Non-Metric Space Library.
    Implements Hierarchical Navigable Small World Graph (HNSW) algorithms.
    HNSW algorithms are one of the fastest and most accurate approximate NN algos.

    Drawbacks:
    NMSLIB is a C++ library with Python bindings.
    It might be problematic to install on some systems.

    For detailed documentation see
    https://github.com/nmslib/nmslib/blob/master/manual/latex/manual.pdf
    """

    _index: NMSLIBFloatIndex
    _metadata: pd.DataFrame
    _dimensionality: int
    _method: str
    _distance: str

    @staticmethod
    def build(data, metadata=None, method="hnsw", distance="l2", print_progress=True):

        assert (
            distance in NMSLIBIndex.AVAILABLE_DISTANCES
        ), f"distance should be one of {NMSLIBIndex.AVAILABLE_DISTANCES}"
        metadata = Index._get_valid_metadata(data, metadata)
        dimensionality = data.shape[1]
        _index = nmslib.init(method=method, space=distance)
        _index.addDataPointBatch(data)
        _index.createIndex(print_progress=print_progress)
        return NMSLIBIndex(_index, metadata, dimensionality, method, distance)

    def find_similar_raw(
        self, query_object: np.ndarray, n_returned: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.validate_input_data(query_object)
        return self._index.knnQuery(query_object, k=n_returned)

    def metadata(self):
        return self._metadata

    def dimensionality(self):
        return self._dimensionality

    def _get_config(self):
        return {
            "_method": self._method,
            "_distance": self._distance,
            "_dimensionality": self._dimensionality,
        }

    def _save_index(self, index_path):
        self._index.saveIndex(str(index_path))

    @classmethod
    def _load_from_disk(cls, index_path, config, metadata):
        index = nmslib.init(method=config["_method"], space=config["_distance"])
        index.loadIndex(str(index_path))
        return NMSLIBIndex(index, metadata, **config)

    AVAILABLE_DISTANCES = [
        "bit_hamming",
        "bit_jaccard",
        "jaccard_sparse",
        "l1",
        "l1_sparse",
        "l2",
        "l2_sparse",
        "linf",
        "linf_sparse",
        "lp:p=",
        "lp_sparse:p=",
        "angulardist",
        "angulardist_sparse",
        "angulardist_sparse_fast",
        "jsmetrslow",
        "jsmetrfast",
        "jsmetrfastapprox",
        "leven",
        "sqfd_minus_func",
        "jsdivslow",
        "jsdivfast",
        "jsdivfastapprox",
        "cosinesimil",
        "cosinesimil_sparse",
        "cosinesimil_sparse_fast",
        "normleven",
        # Non-metricspaces(non-symmetricdistance)
        "kldivfast",
        "kldivfastrq",
        "kldivgenslow",
        "kldivgenfast",
        "kldivgenfastrq",
        "itakurasaitoslow,itakurasaitofast",
        "itakurasaitofastrq",
        "renyidiv_slow",
        "renyidiv_fast",
    ]
