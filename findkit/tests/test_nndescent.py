from findkit.index import NNDescentIndex
from .test_utils import fake_random_data, fake_query


def test_pynndescent_index_setup(fake_random_data):

    index = NNDescentIndex.build(fake_random_data)

    assert True


def test_pynndescent_index_query(fake_query, fake_random_data):

    index = NNDescentIndex.build(fake_random_data)

    n_neighbors = 5
    ids, distances = index.find_similar(fake_query.reshape(1, -1), n_neighbors)

    assert len(ids) == n_neighbors and len(distances) == n_neighbors
