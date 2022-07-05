import numpy as np
from findkit.index import NMSLIBIndex
from .test_utils import fake_random_data, fake_query


def test_nmslib_index_setup(fake_random_data):

    index = NMSLIBIndex.build(fake_random_data)

    assert index.dimensionality() == fake_random_data.shape[1]


def test_nmslib_index_query(fake_query, fake_random_data):

    index = NMSLIBIndex.build(fake_random_data)

    n_neighbors = 5
    ids, distances = index.find_similar_raw(fake_query, n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors


def test_nmslib_index_save_load(fake_query, fake_random_data):
    index = NMSLIBIndex.build(fake_random_data)
    path = "/tmp/nmslib_index"
    index.save(path)

    new_index = NMSLIBIndex.load(path)
    assert new_index.dimensionality() == index.dimensionality()
    assert new_index._method == index._method
    assert new_index._distance == index._distance
