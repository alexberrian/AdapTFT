import numpy as np
from audio_io import AudioSignal
#from typing import Callable
#from typing import Generator


class TFTransformer(object):
    def __init__(self, filename):
        self.AudioSignal = AudioSignal(filename)
        self.param_dict = {}

        self.initialize_default_params()

    def initialize_default_params(self):
        self.param_dict = {"hopsize":                       256,  # Test with 215
                           "windowfunc":             np.hanning,
                           "windowmode":               "single",
                           "windowsize":                   4096,  # Test with 4095
                           "fftsize":                      4096,
                           "buffermode":    "centered_analysis",
                           "sstsize":                      6000,  # Test with 6007
                           "realfft":                      True,
                           "compute_stft":                 True,
                           "compute_sst":                 False,
                           "compute_jtfrm":               False,
                           "compute_qstft":               False,
                           "eps_division":              1.0e-16,
                           "reassignment_mode":         "magsq",  # "magsq" or "complex"
                          }

    def compute_stft(self):
        """
        Computes STFT and returns it as a generator with each STFT frame.
        Allows for support of boundary frames.

        TBD:
        - Proper boundary treatment to ensure perfect reconstruction
        - Option for stereo->mono for computation
        - Testing for stereo signals

        :yield: Generator, each instance is an STFT frame.
        """
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

        # Compute the left boundary STFT frames
        # Will refactor later when I put in the "reconstruction" buffering mode.
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then yield the boundary STFT frame
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[frame0:], windowsize, 'left')
                yield self.wft(reflect_block, window, fftsize)
                frame0 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame0 = 0
        elif buffermode == "valid_analysis":
            frame0 = 0
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
            yield self.wft(block, window, fftsize)
            frame0 += hopsize

        # Compute the right boundary STFT frames
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            final_block = self.AudioSignal.read()  # Read the rest of the file from there
            final_block = final_block.T
            final_frames = final_block.shape[0]
            if final_frames >= windowsize:
                raise ValueError("You shouldn't have final_frames {} "
                                 "greater than windowsize {}".format(final_frames, windowsize))
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame1 = 0
            halfwindowsize = (windowsize // 2)  # Floored if odd
            while final_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[frame1:], windowsize, 'right')
                yield self.wft(reflect_block, window, fftsize)
                frame1 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame1 = 0
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

    def compute_sst(self):
        windowmode = self.param_dict["windowmode"]
        if windowmode != "single":
            raise ValueError("windowmode (currently) must be 'single' for SST, "
                             "instead it is {}. "
                             "Will support SST based on QSTFT later.".format(windowmode))
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        if windowsize > fftsize:
            raise ValueError("window size {} is larger than FFT size {}!".format(windowsize, fftsize))
        sstsize = self.param_dict["sstsize"]
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        overlap = windowsize - hopsize
        window = windowfunc(windowsize)
        eps_division = self.param_dict["eps_division"]
        reassignment_mode = self.param_dict["reassignment_mode"]

        twopi = np.pi * 2
        channels = self.AudioSignal.channels
        if channels > 1:
            sstshape = (sstsize, channels)
        else:
            sstshape = (sstsize, )
        if reassignment_mode == "magsq":
            rmvaluemap = lambda x: np.abs(x) ** 2.0
        elif reassignment_mode == "complex":
            rmvaluemap = lambda x: x
        else:
            raise ValueError("Invalid reassignment_mode {}".format(reassignment_mode))

        # Compute the left boundary SST frames
        # Will refactor later when I put in the "reconstruction" buffering mode.
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize + 1)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then yield the boundary STFT frames necessary
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[frame0:(frame0 + windowsize)],
                                                        windowsize, 'left')
                wft = self.wft(reflect_block, window, fftsize)
                reflect_block = self._pad_boundary_rows(initial_block[(frame0 + 1):(frame0 + 1 + windowsize)],
                                                        windowsize, 'left')
                wft_plus = self.wft(reflect_block, window, fftsize)
                rf = np.angle(wft_plus / (wft + eps_division)) / twopi   # Unit: Normalized frequency
                yield np.add.at(np.zeros(sstshape), (rf * sstsize).astype(int), rmvaluemap(wft))
                frame0 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame0 = 0
        elif buffermode == "valid_analysis":
            frame0 = 0
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

    def wft(self, block: np.ndarray, window: np.ndarray, fftsize: int) -> np.ndarray:
        if self.param_dict["realfft"]:
            return np.fft.rfft(self._zeropad_rows(window * block, fftsize))
        else:
            return np.fft.fft(self._zeropad_rows(window * block, fftsize))

    @staticmethod
    def _pad_boundary_rows(input_array: np.ndarray, finalsize: int, side: str) -> np.ndarray:
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

    @staticmethod
    def _zeropad_rows(input_array: np.ndarray, finalsize: int) -> np.ndarray:
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
