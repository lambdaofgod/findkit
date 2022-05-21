from findkit.vector_loader import STFTVectorLoader
import numpy


def test_load_from_files():

    import librosa

    audio_path = librosa.util.example_audio_file()  # Kevin_MacLeod_-_Vibe_Ace.ogg

    sampling_rate = 22050
    loader = STFTVectorLoader(sampling_rate=sampling_rate)

    m = loader.from_files([audio_path])
    assert type(m) is numpy.ndarray
    assert m.shape == (1, 513, 2647)
