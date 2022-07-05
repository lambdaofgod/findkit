import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple


class Index(ABC):
    def find_similar(self, query_object: np.ndarray, n_returned: int) -> pd.DataFrame:
        """
        Perform nearest neighbor query on index
        and return relevant metadata

        Parameters
        ----------
        query_object : numpy array (of shape (dimensionality,))
            object for which nearest neighbors are found

        n_returned : int
            number of returned most similar objects


        Returns
        -------
        metadata : pandas dataframe containing metadata on items with appended distances from query
        """
        indices, distances = self.find_similar_raw(query_object, n_returned)
        results = self.metadata().iloc[indices]
        distances = pd.Series(distances, name="distance", index=results.index)
        return pd.concat([results, distances], axis=1)

    @abstractmethod
    def find_similar_raw(
        self, query_object: np.ndarray, n_returned: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform nearest neighbor query on index,
        but return only information of retrieved items' index positions and distance

        Parameters
        ----------
        query_object : numpy array (of shape (dimensionality,))
            object for which nearest neighbors are found

        n_returned : int
            number of returned most similar objects


        Returns
        -------
        indices : iterable of int
            indices of nearest neighbors

        distances : iterable of float
            distances between query_object and nearest neighbors

        """

    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def dimensionality(self) -> int:
        pass

    @classmethod
    def _check_metadata_consistency(cls, data, metadata):
        return data.shape[0] == metadata.shape[0]

    def validate_input_data(self, query_object):
        query_shape = query_object.shape
        dim = self.dimensionality()
        assert query_shape == (
            dim,
        ), f"shape of query {query_shape} != {(dim, )} shape of data "

    @classmethod
    def _get_valid_metadata(cls, data, metadata):
        if metadata is None:
            metadata = pd.DataFrame({"i": np.arange(data.shape[0])})
        assert type(metadata) is pd.DataFrame, "metadata should be a pandas DataFrame"
        assert cls._check_metadata_consistency(
            data, metadata
        ), "metadata shape should match data shape"
        return metadata
