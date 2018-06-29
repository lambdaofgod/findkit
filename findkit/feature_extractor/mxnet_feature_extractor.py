from .feature_extractor import FeatureExtractor
from mxnet.module import Module
from mxnet.io import NDArrayIter


class MXNetFeatureExtractor(FeatureExtractor):

    def __init__(self, module, layer_name=None):
        if layer_name is not None:
            module = self.truncate_module(module, layer_name)
        else:
            module = module
        self.transformer = module

    def extract_features(self, data, **kwargs):
        data_iter = NDArrayIter(data)
        return self.transformer.predict(data_iter).asnumpy()

    @classmethod
    def truncate_module(cls, module, layer_name):
        intermediate_layer = module.symbol.get_internals()[layer_name + '_output']
        intermediate_module = Module(intermediate_layer, label_names=[])
        intermediate_module.bind(data_shapes=module.data_shapes)
        intermediate_module.init_params(arg_params=module.get_params()[0])
        return intermediate_module
