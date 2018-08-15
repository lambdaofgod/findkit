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

Load data -> Extract features -> Construct Nearest Neighbor Index

**Tools used in pipeline**:

Numpy -> `FeatureExtractor` -> `Index`

### What's implemented

- `FeatureExtractor`
    - `KerasFeatureExtractor` ([minimal example](https://github.com/lambdaofgod/findkit/blob/master/examples/keras%20extractor%20%26%20annoy%20index.ipynb))
    - `SklearnFeatureExtractor`([minimal example](https://github.com/lambdaofgod/findkit/blob/master/examples/sklearn%20extractor%20%26%20annoy%20index.ipynb))
    
- `Index`
    - `AnnoyIndex`