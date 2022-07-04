from dataclasses import dataclass

import pandas as pd

from ..index.index import Index

try:
    import annoy

    AnnoyIndexImpl = annoy.AnnoyIndex
except ModuleNotFoundError:
    AnnoyIndexImpl = "Annoy not found"


@dataclass(frozen=True)
class AnnoyIndex(Index):

    _index: AnnoyIndexImpl
    _num_examples: int
    _metric: str
    _metadata: pd.DataFrame
    _dimensionality: int

    @staticmethod
    def build(data, metadata=None, n_trees=25, metric="euclidean"):

        metadata = Index._get_valid_metadata(data, metadata)
        dimensionality = data.shape[1]
        num_examples = data.shape[0]
        metric = metric
        _index = annoy.AnnoyIndex(f=dimensionality, metric=metric)
        for i, item in enumerate(data):
            _index.add_item(i, item)
        _index.build(n_trees)
        this = AnnoyIndex(_index, num_examples, metric, metadata, dimensionality)
        return this

    def find_similar_raw(self, query_object, n_returned):
        self.validate_input_data(query_object)
        return self._index.get_nns_by_vector(
            query_object, n=n_returned, include_distances=True
        )

    def metadata(self):
        return self._metadata

    def dimensionality(self):
        return self._dimensionality
