import pandas as pd
import pytest
from findkit.index import InMemoryBM25Index

from .test_utils import fake_query, fake_random_data


@pytest.fixture
def corpus():
    return ["metric learning", "learning to rank", "artificial life"]


def test_bm25_index_setup(corpus):

    index = InMemoryBM25Index.build(corpus, pd.DataFrame({"text": corpus}))

    assert len(index.metadata()) == 3


def test_bm25_index_query(corpus):
    index = InMemoryBM25Index.build(corpus, pd.DataFrame({"text": corpus}))
    n_neighbors = 3
    ids, distances = index.find_similar_raw("metric learning", n_neighbors)
    assert len(ids) == n_neighbors and len(distances) == n_neighbors
    assert list(ids) == [0,1,2]


# def test_bm25_index_save_load(fake_query, fake_random_data):
#    index = BM25Index.build(fake_random_data)
#    path = "/tmp/bm25_index"
#    index.save(path)
#
#    new_index = BM25Index.load(path)
#    assert new_index.dimensionality() == index.dimensionality()
#    assert new_index._method == index._method
#    assert new_index._distance == index._distance
