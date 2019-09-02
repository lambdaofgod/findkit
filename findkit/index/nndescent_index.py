from ..index.index import Index
import pynndescent
import attr


@attr.s
class NNDescentIndex(Index):

    _index: pynndescent.NNDescent = attr.ib()

    @staticmethod
    def build(data, **kwargs):
        nnd_index = pynndescent.NNDescent(data, **kwargs)
        return NNDescentIndex(nnd_index)

    def find_similar(self, query_object, n_returned):
        indices, dists = self._index.query(query_object.reshape(1, -1), k=n_returned)
        return indices.reshape(-1), dists.reshape(-1)