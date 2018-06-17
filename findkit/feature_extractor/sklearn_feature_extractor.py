from .feature_extractor import FeatureExtractor


class SklearnFeatureExtractor(FeatureExtractor):

    def __init__(self, sklearn_transformer):
        self.transformer = sklearn_transformer

    def extract_features(self, data):
        return self.transformer.transform(data)
