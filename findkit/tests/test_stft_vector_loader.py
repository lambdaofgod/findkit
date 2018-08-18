from findkit.vector_loader import STFTVectorLoader
import librosa
import numpy


def test_load_from_files():
    audio_path = librosa.util.example_audio_file()

    sampling_rate = 22050
    loader = STFTVectorLoader(sampling_rate=sampling_rate)

    m = loader.from_files([audio_path])
    assert type(m) is numpy.ndarray
    assert m.shape == (1, 513, 2647)