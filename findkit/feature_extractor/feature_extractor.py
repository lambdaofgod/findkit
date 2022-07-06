from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
    wrapped model to transform data
    """

    @abstractmethod
    def extract_features(self, data, **kwargs):
        """
        Parameters
        ----------
        data : numpy array


        Returns
        -------
        transformed_data : numpy array of shape (shape (data.shape[0], dimensionality))
        """
        pass
