import numpy as np
from findkit.index import NMSLIBIndex
from .test_utils import fake_random_data, fake_query


def test_nmslib_index_setup(fake_random_data):

    index = NMSLIBIndex.build(fake_random_data)

    assert True


def test_nmslib_index_query(fake_query, fake_random_data):

    index = NMSLIBIndex.build(fake_random_data)

    n_neighbors = 5
    ids, distances = index.find_similar_raw(fake_query, n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors
