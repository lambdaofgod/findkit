import numpy as np
from findkit.index import NMSLIBIndex
from .test_utils import fake_random_data


def test_nmslib_index_setup(fake_random_data):

    index = NMSLIBIndex(fake_random_data)

    assert True


def test_nmslib_index_query(fake_random_data):

    index = NMSLIBIndex(fake_random_data)

    n_neighbors = 5
    ids, distances = index.find_similar(np.ones(5), n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors
