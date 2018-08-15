from .feature_extractor import FeatureExtractor


class SklearnFeatureExtractor(FeatureExtractor):
    """
    Wrapper for scikit-learn transformers

    Parameters
    ----------
    sklearn_transformer: object with `transform` or with `fit_transform` method
    """

    def __init__(self, sklearn_transformer):
        self.transformer = sklearn_transformer

    def extract_features(self, data, **kwargs):
        return self.transformer.transform(data)

    def fit_extract_features(self, data, **kwargs):
        return self.transformer.fit_transform(data)
