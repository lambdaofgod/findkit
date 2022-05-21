import pytest
from .test_utils import fake_random_data
import numpy as np

from keras.layers import Dense, Input, Activation
from keras.models import Model

from findkit.feature_extractor import KerasFeatureExtractor


@pytest.fixture
def keras_model():
    input_tensor = Input(shape=(11,))

    hid_z = Dense(1)(input_tensor)
    hid_a = Activation("sigmoid", name="hidden_layer")(hid_z)

    mlp_model = Model(inputs=[input_tensor], outputs=[hid_a])
    mlp_model.compile(optimizer="adam", loss="mse")
    return mlp_model


@pytest.fixture
def fake_target():
    return np.random.randn(100)


def test_keras_feature_extractor(fake_random_data, fake_target, keras_model):

    keras_model.fit(fake_random_data, fake_target)
    feature_extractor = KerasFeatureExtractor(keras_model, "hidden_layer")

    features = feature_extractor.extract_features(fake_random_data)
    assert features.shape == (100, 1)
