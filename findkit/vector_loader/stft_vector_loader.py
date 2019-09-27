from .function_vector_loader import FunctionVectorLoader


try:
    import librosa
except:
    import logging
    logging.info('Warning: you did not install librosa')


class STFTVectorLoader(FunctionVectorLoader):
    """
    Load vectors from audio files using librosa for loading and STFT extractioin
    Handles .wav, .mp3 and .ogg formats if appropriate codecs are installed.

    Parameters
    ----------

    sampling_rate: int
        Sampling rate used by audio files

    n_fft: int
    hop_length: int
    window: scipy window function, or str
        STFT parameters, see https://librosa.github.io/librosa/generated/librosa.core.stft.html

    res_type: string
        librosa loader parameter, see https://librosa.github.io/librosa/generated/librosa.core.load.html
    """

    def __init__(self, sampling_rate, n_fft=1024, hop_length=512, window='hann', res_type='kaiser_fast'):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.res_type = res_type

    def _load_vector(self, file_path):
        return self._get_stft(self._load_wave(file_path))

    def _load_wave(self, file_path):
        Y, __ = librosa.load(file_path, sr=self.sampling_rate, res_type=self.res_type)
        return Y

    def _get_stft(self, wave):
        return librosa.stft(wave, n_fft=self.n_fft, window=self.window, hop_length=self.hop_length)

    def _validate_path(self, file_path):
        return True
