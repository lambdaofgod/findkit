import attr

from ..index.index import Index


@attr.s
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

    _index = attr.ib()
    _metadata = attr.ib()
    _dim = attr.ib()

    @staticmethod
    def build(data, metadata=None, method="hnsw", distance="l2", print_progress=True):
        import nmslib

        assert distance in NMSLIBIndex.AVAILABLE_DISTANCES
        metadata = Index._get_valid_metadata(data, metadata)
        dimensionality = data.shape[1]
        _index = nmslib.init(method=method, space=distance)
        _index.addDataPointBatch(data)
        _index.createIndex(print_progress=print_progress)
        return NMSLIBIndex(_index, metadata, dimensionality)

    def find_similar_raw(self, query_object, n_returned):
        assert query_object.shape[0] == self.get_dimensionality()
        return self._index.knnQuery(query_object, k=n_returned)

    def get_dimensionality(self):
        return self._dim

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
