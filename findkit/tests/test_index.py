import pytest
import attr
from findkit.index.index import Index
import pandas as pd


@attr.s
class MockIndex(Index):

    _metadata = attr.ib()

    def find_similar_raw(self, query_object, n_returned):
        return list(range(n_returned)), [0]


def test_metadata_filter():

    metadata = pd.DataFrame({"name": ["foo", "bar"], "x": [1, 2]})

    mock_index = MockIndex(metadata)
    query_result = mock_index.find_similar(None, 1)
    expected_query_result = metadata.iloc[:1].copy()
    expected_query_result["distance"] = 0

    assert query_result.equals(expected_query_result)
