import numpy as np
import pandas as pd


class Index:
    def find_similar(self, query_object, n_returned):
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
        metadata = self.get_metadata()
        indices, distances = self.find_similar_raw(query_object, n_returned)
        results = metadata.iloc[indices]
        distances = pd.Series(distances, name="distance", index=results.index)
        return pd.concat([results, distances], axis=1)

    def find_similar_raw(self, query_object, n_returned):
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
        raise NotImplementedError()

    def get_metadata(self) -> pd.DataFrame:
        return self._metadata

    @classmethod
    def _check_metadata_consistency(cls, data, metadata):
        return data.shape[0] == metadata.shape[0]

    @classmethod
    def _get_valid_metadata(cls, data, metadata):
        if metadata is None:
            metadata = pd.DataFrame({"i": np.arange(data.shape[0])})
        assert type(metadata) is pd.DataFrame, "metadata should be a pandas DataFrame"
        assert cls._check_metadata_consistency(
            data, metadata
        ), "metadata shape should match data shape"
        return metadata

    def get_dimensionality(self) -> int:
        raise NotImplementedError()
