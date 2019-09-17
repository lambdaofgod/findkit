import attr

from ..index.index import Index


@attr.s
class AnnoyIndex(Index):
    import annoy

    _index: annoy.AnnoyIndex = attr.ib()
    _metadata = attr.ib()
    dimensionality: int = attr.ib()
    num_examples: int = attr.ib()
    metric: str = attr.ib()

    @staticmethod
    def build(data, metadata=None, n_trees=25, metric='euclidean'):
        import annoy
        metadata = Index._get_valid_metadata(data, metadata)
        dimensionality = data.shape[1]
        num_examples = data.shape[0]
        metric = metric
        _index = annoy.AnnoyIndex(f=dimensionality, metric=metric)
        for i, item in enumerate(data):
            _index.add_item(i, item)
        _index.build(n_trees)
        this = AnnoyIndex(_index, metadata, dimensionality, num_examples, metric)
        return this

    def find_similar_raw(self, query_object, n_returned):
        return self._index.get_nns_by_vector(query_object, n=n_returned, include_distances=True)

