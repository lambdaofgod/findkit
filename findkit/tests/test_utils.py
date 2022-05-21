import pytest
import numpy as np


@pytest.fixture
def fake_random_data():
    data_2d = np.random.randn(100, 1)
    return np.hstack([data_2d, np.zeros((100, 10))])


@pytest.fixture
def fake_query():
    return np.random.randn(11)
