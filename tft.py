import numpy as np
from audio_io import AudioSignal
from typing import Callable

def zeropad_rows(input_array: np.ndarray, finalsize: int):
    """
    Note that this expects ROW inputs to be padded...
    you need to transpose the blocks coming in from the block reader.

    :param input_array:
    :param finalsize:
    :return:
    """
    padsize_1 = finalsize // 2
    padsize_2 = finalsize - padsize_1

    if len(input_array.shape) > 1:
        output_array = np.pad(input_array, ((padsize_1, padsize_2), (0, 0)), mode='constant')
    else:
        output_array = np.pad(input_array, (padsize_1, padsize_2), mode='constant')

    return output_array


class TFTransformer(object):
    def __init__(self, filename):
        self.AudioSignal = AudioSignal(filename)
        self.param_dict = {}

        self.initialize_default_params()

    def initialize_default_params(self):
        self.param_dict = {"hopsize":           256,
                           "windowfunc": np.hanning,
                           "windowmode":   "single",
                           "windowsize":       4096,
                           "fftsize":          4096,
                           "sstsize":          6000,
                           "realfft":          True,
                           "compute_stft":     True,
                           "compute_sst":     False,
                           "compute_jtfrm":   False,
                           "compute_qstft":   False,
                          }

    def compute_stft(self):
        windowmode = self.param_dict["windowmode"]
        if windowmode != "single":
            raise ValueError("windowmode must be 'single' for STFT, "
                             "instead it is {}".format(windowmode))
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        windowfunc = self.param_dict["windowfunc"]

        overlap = windowsize - hopsize
        blockreader = self.AudioSignal.blocks(blocksize=windowsize, overlap=overlap)

        window = windowfunc[windowsize]
        stft = []
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            stft.append(self.wft(block, window, fftsize))
        return stft

    def compute_sst(self):
        pass

    def wft(self, block: np.ndarray, window: np.ndarray, fftsize: int):
        if self.param_dict["realfft"]:
            return np.fft.rfft(zeropad_rows(window * block, fftsize))
        else:
            return np.fft.fft(zeropad_rows(window * block, fftsize))
