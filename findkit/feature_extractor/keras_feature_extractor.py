from .feature_extractor import FeatureExtractor
from keras.models import Model


class KerasFeatureExtractor(FeatureExtractor):
    """
    Wrapper for Keras models

    Parameters
    ----------
    keras_model: keras.models.Model

    layer_name: str, optional
        Name of layer used for extracting features. Uses last layer by default
    """

    def __init__(self, keras_model, layer_name=None):
        if layer_name is not None:
            model = self.truncate_model(keras_model, layer_name)
        else:
            model = keras_model
        self.transformer = model

    def extract_features(self, data, **kwargs):
        batch_size = kwargs.get('batch_size')
        return self.transformer.predict(data, batch_size=batch_size)

    @classmethod
    def truncate_model(cls, keras_model, layer_name):
        intermediate_layer = keras_model.get_layer(name=layer_name)
        return Model(inputs=[keras_model.input], outputs=[intermediate_layer.output])
