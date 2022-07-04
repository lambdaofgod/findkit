from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, data, **kwargs):
        """
        wrapped model to transform data

        Parameters
        ----------
        data : numpy array


        Returns
        -------
        transformed_data : numpy array of shape (shape (data.shape[0], dimensionality))
        """
        pass
