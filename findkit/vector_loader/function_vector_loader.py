from .vector_loader import VectorLoader
import numpy as np


class FunctionVectorLoader(VectorLoader):
    """
    Load vectors from files using specified function, then concatenate it into matrix
    """

    def from_files(self, file_paths):
        vectors = [
            self._load_vector(file_path)
            for file_path in file_paths
            if self._validate_path(file_path)
        ]
        return np.stack(vectors)

    def _load_vector(self, file_path):
        raise NotImplementedError

    def _validate_path(self, file_path):
        raise NotImplementedError
