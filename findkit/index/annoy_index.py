from ..index.index import Index


class AnnoyIndex(Index):

    def __init__(self, data, n_trees, metric='euclidean'):
        import annoy

        self.dimensionality = data.shape[1]
        self.num_examples = data.shape[0]
        self.metric = metric
        self._index = annoy.AnnoyIndex(f=self.dimensionality, metric=metric)

        self._build_index(data, n_trees)

    def _build_index(self, data, n_trees):
        for i, item in enumerate(data):
            self._index.add_item(i, item)

        self._index.build(n_trees)

    def find_similar(self, query_object, n_returned):
        return self._index.get_nns_by_vector(query_object, n=n_returned, include_distances=True)

