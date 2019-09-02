import attr

from ..index.index import Index


@attr.s
class NMSLIBIndex(Index):
    _index = attr.ib()

    @staticmethod
    def build(data, method='hnsw', metric='l2', print_progress=True):
        import nmslib
        _index = nmslib.init(method=method, space=metric)
        _index.addDataPointBatch(data)
        _index.createIndex(print_progress=print_progress)
        return NMSLIBIndex(_index)

    def find_similar(self, query_object, n_returned):
        return self._index.knnQuery(query_object, k=n_returned)
