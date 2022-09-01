from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")


class FeatureExtractor(ABC, Generic[T]):
    """
    wrapped model to transform data
    """

    @abstractmethod
    def extract_features(self, data: T, **kwargs):
        """
        Parameters
        ----------
        data : type depends on context - some extractors work on strings, other on numpy arrays
        the only restriction here is that data is supposed to fit in memory

        Returns
        -------
        transformed_data : numpy array of shape (shape (len(data), dimensionality))
        """

    def save(self, path):
        pass

    def load(self, path):
        pass
