import attr

from ..index.index import Index


@attr.s
class AnnoyIndex(Index):

    _index = attr.ib()
    dimensionality = attr.ib()
    num_examples = attr.ib()
    metric = attr.ib()

    @staticmethod
    def build(data, n_trees, metric='euclidean'):
        import annoy
        dimensionality = data.shape[1]
        num_examples = data.shape[0]
        metric = metric
        _index = annoy.AnnoyIndex(n_trees=n_trees, f=dimensionality, metric=metric)
        return AnnoyIndex(_index, dimensionality, num_examples, metric)

    def _build_index(self, data, n_trees):
        for i, item in enumerate(data):
            self._index.add_item(i, item)

        self._index.build(n_trees)

    def find_similar(self, query_object, n_returned):
        return self._index.get_nns_by_vector(query_object, n=n_returned, include_distances=True)

