# findkit

----

A Python library for content-based information retrieval.

### Goal

Provide utilities for easy setup of CBIR systems.

### Summary

Modern deep learning models can often be used for **extracting features** from different types of data - for example images and music.

On the other hands there are methods for **similarity search** based on k-nearest neighbors algorithms.

This library aims to provide unified interface for machine learning frameworks and nearest-neighbors indexing libraries, and to bridge the gap between them.

#### Main pipeline

![](resources/Pipeline%20Diagram.png)

### What's implemented

- `VectorLoader`
    - `FunctionVectorLoader`
        - Audio
            - `STFTVectorLoader` (uses [librosa](https://librosa.github.io/librosa/))
            - `EssentiaVectorLoader` (uses features extracted using [essentia](http://essentia.upf.edu/documentation/), planned) 
    - `Doc2VecLoader` (planned)

- `FeatureExtractor`
    - `KerasFeatureExtractor` ([minimal example](https://github.com/lambdaofgod/findkit/blob/master/examples/keras%20extractor%20%26%20annoy%20index.ipynb))
    - `SklearnFeatureExtractor`([minimal example](https://github.com/lambdaofgod/findkit/blob/master/examples/sklearn%20extractor%20%26%20annoy%20index.ipynb))
    - `MXNetFeatureExtractor` (uses [MXNet Module API](https://mxnet.apache.org/api/python/module/module.html), unfinished)
    - `GluonFeatureExtractor` (planned)
    
- `Index`
    - `AnnoyIndex`
    - `NMSLibIndex` (planned)
    
    
### Useful links

* [Notes on Music Information Retrieval](https://musicinformationretrieval.com)
* [Keras models](https://keras.io/applications/)
* [MXNet](https://mxnet.apache.org)
* [annoy](https://github.com/spotify/annoy)
* [NMSLib](https://github.com/nmslib/nmslib/tree/master/python_bindings), its [manual](https://pdfs.semanticscholar.org/d9d8/744fa1c527780739a843fd825b669a372a24.pdf) is a good source of knowledge on approximate nearest neighbors algorithms
* [Approximate Nearest Neighbours for Recommender Systems](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/)
