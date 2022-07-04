import numpy as np
from findkit.index import AnnoyIndex
from .test_utils import fake_random_data, fake_query


def test_annoy_index(fake_random_data):

    index = AnnoyIndex.build(fake_random_data, n_trees=10)

    n_examples, dimensionality = fake_random_data.shape
    assert index.dimensionality() == dimensionality
    assert index._num_examples == n_examples


def test_annoy_index_query(fake_query, fake_random_data):

    index = AnnoyIndex.build(fake_random_data, n_trees=10)

    n_neighbors = 5
    ids, distances = index.find_similar_raw(fake_query, n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors
