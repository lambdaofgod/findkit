import pytest
from findkit import feature_extractor, retrieval_pipeline, index, serialization
import numpy as np
from sklearn import decomposition
from unittest import mock
import pickle


class MockFeatureExtractor(feature_extractor.FeatureExtractor):
    def extract_features(self, data):
        if type(data) is list:
            return data[0]
        else:
            return data


data = np.array([[1, 0, 0], [-1, 1, 0], [0, 1, -1]])
query = np.array([1, 1, 0])


def test_build_retrieval_pipeline():
    feature_extractor = MockFeatureExtractor()
    pipeline_factory = retrieval_pipeline.RetrievalPipelineFactory(
        feature_extractor, index.NMSLIBIndex.build
    )
    pipeline = pipeline_factory.build(data)
    results = pipeline.find_similar(query, 2)
    assert results["i"].to_list() == [0, 2]


def test_build_retrieval_pipeline_with_sklearn_from_config():
    index_type = "NMSLIBIndex"
    feature_config = serialization.FeatureExtractorConfig(
        "SklearnFeatureExtractor", "foo.pkl"
    )
    pipeline_config = serialization.PipelineConfig(feature_config, index_type)

    pca = decomposition.PCA()
    pca.fit(data)
    with mock.patch.object(pickle, "load", return_value=pca):
        with mock.patch("builtins.open", mock.mock_open()):
            pipeline_factory = retrieval_pipeline.RetrievalPipelineFactory.from_config(
                pipeline_config
            )
            pipeline = pipeline_factory.build(data)
