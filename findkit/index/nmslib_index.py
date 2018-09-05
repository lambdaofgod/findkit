from ..index.index import Index


class NMSLIBIndex(Index):

    def __init__(self, data, method='hnsw', metric='l2'):
        import nmslib

        self.metric = metric
        self.method = method
        self._index = nmslib.init(method=method, space=metric)

        self._build_index(data)

    def _build_index(self, data, print_progress=True):
        self._index.addDataPointBatch(data)

        self._index.createIndex(print_progress=print_progress)

    def find_similar(self, query_object, n_returned):
        return self._index.knnQuery(query_object, k=n_returned)
