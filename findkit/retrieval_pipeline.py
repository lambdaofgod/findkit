import numpy as np
from findkit.index import Index
from findkit.feature_extractor import FeatureExtractor
from dataclasses import dataclass
from findkit import serialization
from typing import Callable, Optional


class RetrievalPipelineFactory:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        index_factory: Callable[[np.ndarray], Index],
        query_feature_extractor: Optional[FeatureExtractor] = None,
    ):
        self.feature_extractor = feature_extractor
        self.index_factory = index_factory
        self.query_feature_extractor = (
            query_feature_extractor
            if query_feature_extractor is not None
            else feature_extractor
        )

    def build(self, data, **kwargs):
        index = self.index_factory(
            self.feature_extractor.extract_features(data), **kwargs
        )
        return RetrievalPipeline(index=index, feature_extractor=self.feature_extractor, self.query_feature_extractor)

    @classmethod
    def from_config(cls, config: serialization.PipelineConfig):
        feature_extractor = serialization.setup_feature_extractor(config.feature_config)
        index_factory = serialization.get_index_factory(config.index_type)
        return cls(feature_extractor, index_factory)


@dataclass
class RetrievalPipeline:

    feature_extractor: FeatureExtractor
    index: Index
    query_feature_extractor: FeatureExtractor

    def find_similar(self, query, n_returned: int):
        query_obj = self.query_feature_extractor.extract_features(query)
        return self.index.find_similar(query_obj, n_returned)
