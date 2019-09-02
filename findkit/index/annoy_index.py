import attr

from ..index.index import Index


@attr.s
class AnnoyIndex(Index):
    import annoy

    _index: annoy.AnnoyIndex = attr.ib()
    dimensionality: int = attr.ib()
    num_examples: int = attr.ib()
    metric: str = attr.ib()

    @staticmethod
    def build(data, n_trees, metric='euclidean'):
        import annoy
        dimensionality = data.shape[1]
        num_examples = data.shape[0]
        metric = metric
        _index = annoy.AnnoyIndex(f=dimensionality, metric=metric)
        for i, item in enumerate(data):
            _index.add_item(i, item)
        _index.build(n_trees)
        return AnnoyIndex(_index, dimensionality, num_examples, metric)


    def find_similar(self, query_object, n_returned):
        return self._index.get_nns_by_vector(query_object, n=n_returned, include_distances=True)

