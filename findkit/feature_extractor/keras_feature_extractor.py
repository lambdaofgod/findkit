from .feature_extractor import FeatureExtractor
from keras.models import Model


class KerasFeatureExtractor(FeatureExtractor):

    def __init__(self, keras_model, layer_name=None):
        self.transformer = self.truncate_model(keras_model, layer_name)

    def extract_features(self, data, **kwargs):
        batch_size = kwargs.get('batch_size')
        return self.transformer.predict(data, batch_size=batch_size)

    @classmethod
    def truncate_model(cls, keras_model, layer_name):
        if layer_name is not None:
            intermediate_layer = keras_model.get_layer(name=layer_name)
            return Model(inputs=[keras_model.input], outputs=[intermediate_layer.output])
        else:
            return keras_model
