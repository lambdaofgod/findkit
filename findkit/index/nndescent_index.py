from ..index.index import Index
import pynndescent
import attr
import numpy as np


@attr.s
class NNDescentIndex(Index):

    _index: pynndescent.NNDescent = attr.ib()
    _metadata = attr.ib()

    @staticmethod
    def build(data, metadata=None, **kwargs):
        metadata = Index._get_valid_metadata(data, metadata)
        nnd_index = pynndescent.NNDescent(data, **kwargs)
        return NNDescentIndex(nnd_index, metadata)

    def find_similar_raw(self, query_object, n_returned):
        self._index.rng_state = np.array([42, 42, 42], dtype=np.int64)
        indices, dists = self._index.query(query_object.reshape(1, -1), k=n_returned)
        return indices.reshape(-1), dists.reshape(-1)
