import numpy as np
from sklearn.decomposition import PCA

from findkit.feature_extractor import SklearnFeatureExtractor

from .test_utils import fake_random_data


TOL = 1e-6


def test_sklearn_feature_extractor(fake_random_data):
    pca = PCA(n_components=2)
    pca_projected_data = pca.fit_transform(fake_random_data)

    extractor = SklearnFeatureExtractor(pca)
    extractor_projected_data = extractor.extract_features(fake_random_data)

    assert pca_projected_data.shape == (100, 2)
    assert np.linalg.norm(pca_projected_data - extractor_projected_data) < TOL

