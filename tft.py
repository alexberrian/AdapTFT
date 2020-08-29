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
        if windowsize < 2:
            raise ValueError("windowsize {} must be at least 2".format(windowsize))
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
                reflect_block = self._pad_boundary_rows(initial_block[:frame0], windowsize, 'left')
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
            final_block_num_frames = final_block.shape[0]
            if final_block_num_frames >= windowsize:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize {}".format(final_block_num_frames, windowsize))
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2   # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[frame1:], windowsize, 'right')
                yield self.wft(reflect_block, window, fftsize)
                frame1 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

    def compute_sst(self):
        """
        TO DO:
        - Investigate the non-realfft case
        - Deal with mono vs. stereo etc.
        - Obviously condense repeated code into a generic reassignment modules
        :yield: synchrosqueezing transform of the given STFT frame
        """
        if not self.param_dict["realfft"]:
            raise ValueError("Must have realfft to compute SST, untested otherwise!")
        windowmode = self.param_dict["windowmode"]
        if windowmode != "single":
            raise ValueError("windowmode (currently) must be 'single' for SST, "
                             "instead it is {}. "
                             "Will support SST based on QSTFT later.".format(windowmode))
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        if windowsize < 4:
            raise ValueError("windowsize {} must be at least 4 to deal with edge cases for SST".format(windowsize))
        windowsize_p1 = windowsize + 1
        fftsize = self.param_dict["fftsize"]
        if windowsize > fftsize:
            raise ValueError("window size {} is larger than FFT size {}!".format(windowsize, fftsize))
        sstsize = self.param_dict["sstsize"]
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        overlap = (windowsize + 1) - hopsize  # For SST block procedure
        window = windowfunc(windowsize)
        eps_division = self.param_dict["eps_division"]
        reassignment_mode = self.param_dict["reassignment_mode"]

        twopi = np.pi * 2
        channels = self.AudioSignal.channels
        num_bins_up_to_nyquist = (sstsize // 2) + 1
        if channels > 1:
            sstshape = (num_bins_up_to_nyquist, channels)  # Change for non-real FFT
        else:
            sstshape = (num_bins_up_to_nyquist, )  # Change for non-real FFT
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
                reflect_block = self._pad_boundary_rows(initial_block[:(frame0 + windowsize)],
                                                        windowsize, 'left')
                wft = self.wft(reflect_block, window, fftsize)
                # Note that for the below, we go to (frame0 + windowsize_p1) because the important thing
                # is that the reflection of the function about time 0 must be the same so that you are
                # analyzing the same function.
                reflect_block = self._pad_boundary_rows(initial_block[:(frame0 + windowsize_p1)],
                                                        windowsize, 'left')
                wft_plus = self.wft(reflect_block, window, fftsize)
                rf = np.angle(wft_plus / (wft + eps_division)) / twopi   # Unit: Normalized frequency
                out_of_bounds = np.where((rf < 0) | (rf > 0.5))  # For real valued signals rf > 0.5 is meaningless
                wft[out_of_bounds] = 0
                rf[out_of_bounds] = 0
                sst_out = np.zeros(sstshape, dtype=(complex if reassignment_mode == "complex" else float))
                np.add.at(sst_out, (rf * sstsize).astype(int), rmvaluemap(wft))  # Change for non-real FFT
                yield sst_out
                frame0 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
            frame0 = 0
        elif buffermode == "valid_analysis":
            frame0 = 0
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

        # May refactor the following four non-comment code lines for full generality
        # Get the number of audio frames, and seek to the audio frame given by frame0
        num_audio_frames = self.AudioSignal.get_num_frames_from_and_seek_start(start_frame=frame0)

        # Now calculate the max number of FULL non-boundary SST frames,
        # considering hop size and window size.  Have to modify because taking more frames than usual.
        num_full_sst_frames = 1 + ((num_audio_frames - windowsize_p1) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary SST.
        num_audio_frames_full_sst = (num_full_sst_frames - 1) * hopsize + windowsize_p1

        # Feed blocks to create the non-boundary SST frames, with
        blockreader = self.AudioSignal.blocks(blocksize=windowsize_p1, overlap=overlap,
                                              frames=num_audio_frames_full_sst)
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            wft = self.wft(block[:windowsize], window, fftsize)
            wft_plus = self.wft(block[1:], window, fftsize)
            rf = np.angle(wft_plus / (wft + eps_division)) / twopi  # Unit: Normalized frequency
            out_of_bounds = np.where((rf < 0) | (rf > 0.5))  # For real valued signals rf > 0.5 is meaningless
            wft[out_of_bounds] = 0
            rf[out_of_bounds] = 0
            sst_out = np.zeros(sstshape, dtype=(complex if reassignment_mode == "complex" else float))
            np.add.at(sst_out, (rf * sstsize).astype(int), rmvaluemap(wft))  # Change for non-real FFT
            yield sst_out
            frame0 += hopsize

        # Compute the right boundary SST frames
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            final_block = self.AudioSignal.read()  # Read the rest of the file (length less than windowsize+1)
            final_block = final_block.T
            final_block_num_frames = final_block.shape[0]
            if final_block_num_frames >= windowsize_p1:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize + 1 == {}".format(final_block_num_frames, windowsize_p1))
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2   # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[frame1:], windowsize, 'right')
                wft = self.wft(reflect_block, window, fftsize)
                # EDGE CASE: frame1 + 1 may not be valid index of final_block if halfwindowsize == 1,
                # i.e. if windowsize < 4.  That is why we require windowsize >= 4.
                reflect_block = self._pad_boundary_rows(final_block[frame1 + 1:(frame1 + windowsize_p1)],
                                                        windowsize, 'right')
                wft_plus = self.wft(reflect_block, window, fftsize)
                rf = np.angle(wft_plus / (wft + eps_division)) / twopi  # Unit: Normalized frequency
                out_of_bounds = np.where((rf < 0) | (rf > 0.5))  # For real valued signals rf > 0.5 is meaningless
                wft[out_of_bounds] = 0
                rf[out_of_bounds] = 0
                sst_out = np.zeros(sstshape, dtype=(complex if reassignment_mode == "complex" else float))
                np.add.at(sst_out, (rf * sstsize).astype(int), rmvaluemap(wft))  # Change for non-real FFT
                yield sst_out
                frame1 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
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
        if finalsize == inputsize:
            return input_array
        else:
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
        if inputsize == finalsize:
            return input_array
        else:
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
