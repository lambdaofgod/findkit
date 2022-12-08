from dataclasses import dataclass
from findkit import index, feature_extractor, retrieval_pipeline
import pickle
from findkit import index, feature_extractor


INDEX_TYPES = {"NMSLIBIndex": index.NMSLIBIndex}
try:
    from index import nndescent_index
except:
    pass
else:
    INDEX_TYPES["NNDescentIndex"] = nndescent_index.NNDescentIndex


def get_index_factory(index_type):
    assert index_type in INDEX_TYPES, f"unsupported index type: {index_type}"
    return INDEX_TYPES[index_type].build


@dataclass
class FeatureExtractorConfig:
    feature_extractor_type: str
    feature_extractor_artifact_path: str


@dataclass
class PipelineConfig:

    feature_config: FeatureExtractorConfig
    index_type: str


def setup_feature_extractor(cfg: FeatureExtractorConfig):
    if cfg.feature_extractor_type == "SklearnFeatureExtractor":
        with open(cfg.feature_extractor_artifact_path, "rb") as f:
            sklearn_model = pickle.load(f)
        return feature_extractor.SklearnFeatureExtractor(sklearn_model)
