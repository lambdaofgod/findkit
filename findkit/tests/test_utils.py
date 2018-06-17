import pytest
import numpy as np


@pytest.fixture
def fake_random_data():
    data_2d = np.random.randn(5, 1)
    return np.hstack([data_2d, np.zeros((5, 4))])
