import numpy as np
from findkit.index import AnnoyIndex
from .test_utils import fake_random_data


def test_annoy_index(fake_random_data):

    index = AnnoyIndex(fake_random_data, n_trees=10)

    n_examples, dimensionality = fake_random_data.shape
    assert index.dimensionality == dimensionality
    assert index.num_examples == n_examples


def test_annoy_index_query(fake_random_data):

    index = AnnoyIndex(fake_random_data, n_trees=10)

    n_neighbors = 5
    ids, distances = index.find_similar(np.ones(5), n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors
