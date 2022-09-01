import numpy as np
from findkit.index import Index
from findkit.feature_extractor import FeatureExtractor
from dataclasses import dataclass
from findkit import serialization
from typing import Callable


@dataclass
class RetrievalPipelineFactory:
    feature_extractor: FeatureExtractor
    index_factory: Callable[[np.ndarray], Index]

    def build(self, data, **kwargs):
        index = self.index_factory(
            self.feature_extractor.extract_features(data), **kwargs
        )
        return RetrievalPipeline(index=index, feature_extractor=self.feature_extractor)

    @classmethod
    def from_config(cls, config: serialization.PipelineConfig):
        feature_extractor = serialization.setup_feature_extractor(config.feature_config)
        index_factory = serialization.get_index_factory(config.index_type)
        return cls(feature_extractor, index_factory)


@dataclass
class RetrievalPipeline:

    feature_extractor: FeatureExtractor
    index: Index

    def find_similar(self, query, n_returned: int):
        query_obj = self.feature_extractor.extract_features(query)
        return self.index.find_similar(query_obj, n_returned)
