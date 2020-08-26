import numpy as np
from audio_io import AudioSignal
#from typing import Callable


def pad_boundary_rows(input_array: np.ndarray, finalsize: int, side: str) -> np.ndarray:
    """
    Pad each channel of a buffer, where channels are assumed to be in rows.
    Padding happens at the boundary, by even reflection.

    :param input_array: array to be padded by reflection
    :param finalsize: finalsize: final size of the array (example: window size)
    :param side: "left" or "right" to do the padding.
                 i.e., if "left", then padding is done to the left of the input array.
    :return: output_array: reflection-padded array
    """
    inputsize = input_array.shape[0]
    padsize = finalsize - inputsize
    if side == "left":
        padsize_left = padsize
        padsize_right = 0
    elif side == "right":
        padsize_left = 0
        padsize_right = padsize
    else:
        raise ValueError("Pad side {} to pad_boundary_rows is invalid, "
                         "must be 'left' or 'right'".format(side))

    if len(input_array.shape) == 2:
        output_array = np.pad(input_array, ((padsize_left, padsize_right), (0, 0)), mode='reflect')
    elif len(input_array.shape) == 1:
        output_array = np.pad(input_array, (padsize_left, padsize_right), mode='reflect')
    else:
        raise ValueError("input array to pad_boundary_rows has dimensions {}, "
                         "which is not supported".format(input_array.shape))

    return output_array


def zeropad_rows(input_array: np.ndarray, finalsize: int) -> np.ndarray:
    """
    Zeropad each channel of a buffer, where channels are assumed to be in rows.
    Padding happens with the input array centered, and zeros padded equally on left and right,
    unless finalsize minus inputsize is odd.
    This is used for preparing a windowed array to be sent to an FFT.

    :param input_array: array to be padded with zeros
    :param finalsize: final size of the array (example: FFT size)
    :return: output_array: zero-padded array
    """
    inputsize = input_array.shape[0]

    padsize = finalsize - inputsize
    padsize_left = padsize // 2
    padsize_right = padsize - padsize_left

    if len(input_array.shape) == 2:
        output_array = np.pad(input_array, ((padsize_left, padsize_right), (0, 0)), mode='constant')
    elif len(input_array.shape) == 1:
        output_array = np.pad(input_array, (padsize_left, padsize_right), mode='constant')
    else:
        raise ValueError("input array to zeropad_rows has dimensions {}, "
                         "which is not supported".format(input_array.shape))

    return output_array


class TFTransformer(object):
    def __init__(self, filename):
        self.AudioSignal = AudioSignal(filename)
        self.param_dict = {}

        self.initialize_default_params()

    def initialize_default_params(self):
        self.param_dict = {"hopsize":                    256,  # Test with 215
                           "windowfunc":          np.hanning,
                           "windowmode":            "single",
                           "windowsize":                4096,  # Test with 4095
                           "fftsize":                   4096,
                           "buffermode": "centered_analysis",
                           "sstsize":                   6000,  # Test with 6007
                           "realfft":                   True,
                           "compute_stft":              True,
                           "compute_sst":              False,
                           "compute_jtfrm":            False,
                           "compute_qstft":            False,
                          }

    def compute_stft(self) -> np.ndarray:
        windowmode = self.param_dict["windowmode"]
        if windowmode != "single":
            raise ValueError("windowmode must be 'single' for STFT, "
                             "instead it is {}".format(windowmode))
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        if windowsize > fftsize:
            raise ValueError("window size {} is larger than FFT size {}!".format(windowsize, fftsize))
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        overlap = windowsize - hopsize
        window = windowfunc(windowsize)

        # Initialize the stft
        # This should be a separate function
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize)
            initial_block = initial_block.T
            stft = []
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = pad_boundary_rows(initial_block[frame0:], windowsize, 'left')
                stft.append(self.wft(reflect_block, window, fftsize))
                frame0 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame0 = 0
            stft = []
        elif buffermode == "valid_analysis":
            frame0 = 0
            stft = []
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

        # Get the number of audio frames, and seek to the audio frame given by frame0
        num_audio_frames = self.AudioSignal.get_num_frames_from_and_seek_start(start_frame=frame0)

        # Now calculate the max number of FULL non-boundary STFT frames,
        # considering hop size and window size.
        num_full_stft_frames = 1 + ((num_audio_frames - windowsize) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary STFT.
        num_audio_frames_full_stft = (num_full_stft_frames - 1) * hopsize + windowsize

        # Feed blocks to create the non-boundary STFT frames
        blockreader = self.AudioSignal.blocks(blocksize=windowsize, overlap=overlap,
                                              frames=num_audio_frames_full_stft)
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            stft.append(self.wft(block, window, fftsize))
            frame0 += hopsize
        # BUT what happens to the seek position after going through all those blocks?

        # Now treat the right boundary
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            final_block = self.AudioSignal.read()  # Read the rest of the file from there
            final_block = final_block.T
            final_frames = final_block.shape[0]
            print(final_frames)
            if final_frames >= windowsize:
                raise ValueError("You shouldn't have final_frames {} "
                                 "greater than windowsize {}".format(final_frames, windowsize))
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame1 = 0
            halfwindowsize = (windowsize // 2)  # Floored if odd
            while final_frames - frame1 >= halfwindowsize:
                reflect_block = pad_boundary_rows(final_block[frame1:], windowsize, 'right')
                stft.append(self.wft(reflect_block, window, fftsize))
                frame1 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame1 = 0
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

        return np.asarray(stft)

    def compute_sst(self):
        windowmode = self.param_dict["windowmode"]
        if windowmode != "single":
            raise ValueError("windowmode (currently) must be 'single' for SST, "
                             "instead it is {}. "
                             "Will support SST based on QSTFT later.".format(windowmode))
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        windowfunc = self.param_dict["windowfunc"]

    def wft(self, block: np.ndarray, window: np.ndarray, fftsize: int) -> np.ndarray:
        if self.param_dict["realfft"]:
            return np.fft.rfft(zeropad_rows(window * block, fftsize))
        else:
            return np.fft.fft(zeropad_rows(window * block, fftsize))
