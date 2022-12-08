import pytest
from .test_utils import fake_random_data
from mxnet.io import NDArrayIter
import mxnet as mx
from findkit.feature_extractor import mxnet_feature_extractor

import numpy as np


TOL = 1e-6


@pytest.fixture
def mxnet_module():
    data = mx.sym.var("data")

    x = mx.sym.FullyConnected(data, num_hidden=1, name="x")

    y = mx.sym.LinearRegressionOutput(x, name="y")

    loss = mx.sym.MakeLoss(y)

    module = mx.mod.Module(loss, label_names=["y_label"])
    module.bind([("data", (100, 11))], [("y_label", (100, 1))])
    module.init_params()

    return module


def test_mxnet_feature_extractor(fake_random_data, mxnet_module):

    feature_extractor = mxnet_feature_extractor.MXNetFeatureExtractor(mxnet_module, "x")

    extracted_features = feature_extractor.extract_features(fake_random_data)

    intermediate_symbol = mxnet_module.symbol.get_internals()["x_output"]
    symbol_extracted_features_ndarray = intermediate_symbol.eval(
        data=mx.nd.array(fake_random_data), **mxnet_module.get_params()[0]
    )[0]

    assert (
        np.linalg.norm(extracted_features - symbol_extracted_features_ndarray.asnumpy())
        < TOL
    )
