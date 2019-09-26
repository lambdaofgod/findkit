import attr

from ..index.index import Index


@attr.s
class NMSLIBIndex(Index):
    _index = attr.ib()
    _metadata = attr.ib()

    @staticmethod
    def build(data, metadata=None, method='hnsw', metric='l2', print_progress=True):
        import nmslib
        metadata = Index._get_valid_metadata(data, metadata)
        _index = nmslib.init(method=method, space=metric)
        _index.addDataPointBatch(data)
        _index.createIndex(print_progress=print_progress)
        return NMSLIBIndex(_index, metadata)

    def find_similar_raw(self, query_object, n_returned):
        return self._index.knnQuery(query_object, k=n_returned)
