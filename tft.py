import numpy as np
from audio_io import AudioSignal
from copy import deepcopy
# from typing import Callable
# from typing import Generator

TWOPI = np.pi * 2


class TFTransformer(object):

    def __init__(self, filename):
        self.AudioSignal = AudioSignal(filename)
        self.param_dict = {}
        self.exp_time_shift = None
        self.jtfrt_memory = None
        self.jtfrt_shape = None

        self._initialize_default_params()
        self._initialize_helpers_jtfrt()

    def _initialize_default_params(self):
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

    def _initialize_helpers_jtfrt(self):
        """
        Initialize helper arrays and variables for JTFRT
        :return:
        """
        windowsize = self.param_dict["windowsize"]
        hopsize = self.param_dict["hopsize"]
        sstsize = self.param_dict["sstsize"]
        reassignment_mode = self.param_dict["reassignment_mode"]
        reassignment_dtype: type = complex if reassignment_mode == "complex" else float

        self.exp_time_shift = np.exp(-TWOPI * 1j * np.tile(np.arange(self.param_dict["fftsize"]),
                                                           [self.AudioSignal.channels, 1])
                                     / float(self.AudioSignal.samplerate))
        jtfrt_memory_num_frames = windowsize // hopsize
        channels = self.AudioSignal.channels
        num_bins_up_to_nyquist = (sstsize // 2) + 1
        self.jtfrt_shape = (channels, num_bins_up_to_nyquist)
        self.jtfrt_memory = np.zeros([jtfrt_memory_num_frames, *self.jtfrt_shape], dtype=reassignment_dtype)

    def _check_parameter_validity(self, transform):
        windowmode = self.param_dict["windowmode"]
        windowsize = self.param_dict["windowsize"]
        hopsize = self.param_dict["hopsize"]
        fftsize = self.param_dict["fftsize"]
        buffermode = self.param_dict["buffermode"]
        reassignment_mode = self.param_dict["reassignment_mode"]

        # Validity checks for all transforms
        if hopsize > windowsize:
            raise ValueError("Not allowed to have hopsize {} larger than windowsize {} "
                             "because of the way SoundFile processes chunks".format(hopsize, windowsize))
        if windowsize > fftsize:
            raise ValueError("window size {} is larger than FFT size {}!".format(windowsize, fftsize))
        if buffermode not in ["centered_analysis", "reconstruction", "valid_analysis"]:
            raise ValueError("Invalid buffermode {}".format(buffermode))
        elif buffermode == "reconstruction":
            raise ValueError("Buffermode 'reconstruction' is not yet implemented")

        # Transform-specific checks
        if transform == "stft":
            if windowmode != "single":
                raise ValueError("windowmode must be 'single' for STFT, "
                                 "instead it is {}".format(windowmode))
            if windowsize < 2:
                raise ValueError("windowsize {} must be at least 2 for STFT".format(windowsize))
        elif transform in ["sst", "jtfrt"]:
            if not self.param_dict["realfft"]:
                raise ValueError("Must have realfft to compute SST/JTFRT, untested otherwise!")
            if windowmode != "single":
                raise ValueError("windowmode (currently) must be 'single' for SST/JTFRT, "
                                 "instead it is {}. "
                                 "Will support SST/JTFRT based on QSTFT later.".format(windowmode))
            if windowsize < 4:
                raise ValueError("windowsize {} must be at least 4 to deal with edge cases "
                                 "for SST/JTFRT".format(windowsize))
            if reassignment_mode not in ["magsq", "complex"]:
                raise ValueError("Invalid reassignment mode {}".format(reassignment_mode))
        else:
            raise ValueError("Invalid transform {}".format(transform))

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
        self._check_parameter_validity("stft")

        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        overlap = windowsize - hopsize
        window = windowfunc(windowsize)

        # Just in case the audio signal has already been read out
        self.AudioSignal.seek(frames=0)

        # Compute the left boundary STFT frames
        # Will refactor later when I put in the "reconstruction" buffering mode.
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize, always_2d=True)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then yield the boundary STFT frame
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[:, :frame0], windowsize, 'left')
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
                                              frames=num_audio_frames_full_stft, always_2d=True)
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            yield self.wft(block, window, fftsize)
            frame0 += hopsize

        # Compute the right boundary STFT frames
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            final_block = self.AudioSignal.read(always_2d=True)  # Read the rest of the file from there
            final_block = final_block.T
            final_block_num_frames = final_block.shape[1]
            if final_block_num_frames >= windowsize:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize {}".format(final_block_num_frames, windowsize))
            # Pad the boundary with reflected audio frames,
            # then add boundary STFT frames to the STFT
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2   # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[:, frame1:], windowsize, 'right')
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
        - Deal with QSTFT case
        :yield: synchrosqueezing transform of the given STFT frame
        """
        self._check_parameter_validity("sst")

        # windowmode = self.param_dict["windowmode"]  # For later development
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        sstsize = self.param_dict["sstsize"]
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        windowsize_p1 = windowsize + 1
        overlap = windowsize_p1 - hopsize  # For SST block procedure
        window = windowfunc(windowsize)
        reassignment_mode = self.param_dict["reassignment_mode"]

        # Just in case the audio signal has already been read out
        self.AudioSignal.seek(frames=0)

        # Compute the left boundary SST frames
        # Will refactor later when I put in the "reconstruction" buffering mode.
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize + 1, always_2d=True)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then yield the boundary SST frames necessary
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize)], windowsize, 'left')
                reflect_block_plus = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize_p1)],
                                                             windowsize, 'left')
                yield self._reassign_sst(reflect_block, reflect_block_plus, window, fftsize, sstsize,
                                         reassignment_mode)
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
                                              frames=num_audio_frames_full_sst, always_2d=True)
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            yield self._reassign_sst(block[:, :windowsize], block[:, 1:], window, fftsize, sstsize, reassignment_mode)
            frame0 += hopsize

        # Compute the right boundary SST frames
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            # Read the rest of the file (length less than windowsize+1)
            final_block = self.AudioSignal.read(always_2d=True)
            final_block = final_block.T
            final_block_num_frames = final_block.shape[1]
            if final_block_num_frames >= windowsize_p1:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize + 1 == {}".format(final_block_num_frames, windowsize_p1))
            # Pad the boundary with reflected audio frames,
            # then add boundary SST frames to the SST
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2   # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[:, frame1:], windowsize, 'right')
                reflect_block_plus = self._pad_boundary_rows(final_block[:, frame1 + 1:(frame1 + windowsize_p1)],
                                                             windowsize, 'right')
                yield self._reassign_sst(reflect_block, reflect_block_plus, window, fftsize, sstsize,
                                         reassignment_mode)
                frame1 += hopsize
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

    def compute_jtfrt(self):
        """
        TO DO:
        - Investigate the non-realfft case
        - Deal with mono vs. stereo etc.
        - Deal with QSTFT case
        - Edge case: What happens at the right boundary? It's probably fine, but just check.
        :yield: joint time-frequency reassignment transform of the given STFT frame
        """
        self._check_parameter_validity("jtfrt")
        self._initialize_helpers_jtfrt()

        # windowmode = self.param_dict["windowmode"]  # For later development
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        fftsize = self.param_dict["fftsize"]
        sstsize = self.param_dict["sstsize"]  # i.e., size of frequency axis.  No option to change time axis yet
        windowfunc = self.param_dict["windowfunc"]
        buffermode = self.param_dict["buffermode"]
        windowsize_p1 = windowsize + 1
        overlap = windowsize_p1 - hopsize  # For JTFRT block procedure
        window = windowfunc(windowsize)
        reassignment_mode = self.param_dict["reassignment_mode"]
        reassignment_dtype: type = complex if reassignment_mode == "complex" else float

        # Create a circular buffer of JTFRT frames of size windowsize // hopsize,
        # This may not be enough if windowsize not divided evenly by hopsize, but forget that edge case
        jtfrt_memory_num_frames = windowsize // hopsize
        write_frame = 0
        export_frame = -(jtfrt_memory_num_frames // 2) + 1

        # Just in case the audio signal has already been read out
        self.AudioSignal.seek(frames=0)

        # Compute the left boundary JTFRT frames
        # Will refactor later when I put in the "reconstruction" buffering mode.
        if buffermode == "centered_analysis":
            initial_block = self.AudioSignal.read(frames=windowsize + 1, always_2d=True)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then yield the boundary JTFRT frames necessary
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0. reconstruction imperfect
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize)], windowsize, 'left')
                reflect_block_plus = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize_p1)],
                                                             windowsize, 'left')
                self._reassign_jtfrt(write_frame, reflect_block, reflect_block_plus, window,
                                     fftsize, sstsize, reassignment_mode)
                frame0 += hopsize
                write_frame += 1
                write_frame %= jtfrt_memory_num_frames
                if export_frame > -1:
                    # Export and reset this frame to zeros so it can be added to again
                    # You HAVE to yield a deepcopy or else it will yield a pointer to the memory array.
                    yield deepcopy(self.jtfrt_memory[export_frame])
                    self.jtfrt_memory[export_frame] *= 0
                    export_frame += 1
                    export_frame %= jtfrt_memory_num_frames
                else:
                    export_frame += 1

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

        # Now calculate the max number of FULL non-boundary JTFRT frames,
        # considering hop size and window size.  Have to modify because taking more frames than usual.
        num_full_jtfrt_frames = 1 + ((num_audio_frames - windowsize_p1) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary JTFRT.
        num_audio_frames_full_jtfrt = (num_full_jtfrt_frames - 1) * hopsize + windowsize_p1

        # Feed blocks to create the non-boundary JTFRT frames, with
        blockreader = self.AudioSignal.blocks(blocksize=windowsize_p1, overlap=overlap,
                                              frames=num_audio_frames_full_jtfrt, always_2d=True)
        for block in blockreader:
            block = block.T  # First transpose to get each channel as a row
            self._reassign_jtfrt(write_frame, block[:, :windowsize], block[:, 1:],
                                 window, fftsize, sstsize, reassignment_mode)
            frame0 += hopsize
            write_frame += 1
            write_frame %= jtfrt_memory_num_frames
            if export_frame > -1:
                # Export and reset this frame to zeros so it can be added to again
                # You HAVE to yield a deepcopy or else it will yield a pointer to the memory array.
                yield deepcopy(self.jtfrt_memory[export_frame])
                self.jtfrt_memory[export_frame] *= 0
                export_frame += 1
                export_frame %= jtfrt_memory_num_frames
            else:
                export_frame += 1

        # Compute the right boundary JTFRT frames
        if buffermode == "centered_analysis":
            # Need to read from frame0
            self.AudioSignal.seek(frames=frame0)
            # Read the rest of the file (length less than windowsize+1)
            final_block = self.AudioSignal.read(always_2d=True)
            final_block = final_block.T
            final_block_num_frames = final_block.shape[1]
            if final_block_num_frames >= windowsize_p1:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize + 1 == {}".format(final_block_num_frames, windowsize_p1))
            # Pad the boundary with reflected audio frames,
            # then add boundary JTFRT frames to the JTFRT
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2  # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[:, frame1:], windowsize, 'right')
                reflect_block_plus = self._pad_boundary_rows(final_block[:, frame1 + 1:(frame1 + windowsize_p1)],
                                                             windowsize, 'right')
                self._reassign_jtfrt(write_frame, reflect_block, reflect_block_plus, window,
                                     fftsize, sstsize, reassignment_mode)
                frame1 += hopsize
                write_frame += 1
                write_frame %= jtfrt_memory_num_frames
                if export_frame > -1:
                    # Export and reset this frame to zeros so it can be added to again
                    # You HAVE to yield a deepcopy or else it will yield a pointer to the memory array.
                    yield deepcopy(self.jtfrt_memory[export_frame])
                    self.jtfrt_memory[export_frame] *= 0
                    export_frame += 1
                    export_frame %= jtfrt_memory_num_frames
                else:
                    export_frame += 1
        # TODO: NEED TO FLUSH THE REMAINING BUFFERS!!! ( = 1 less than the total size)
        elif buffermode == "reconstruction":
            pass  # FILL THIS IN
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

    def wft(self, block: np.ndarray, window: np.ndarray, fftsize: int, fft_type="real") -> np.ndarray:
        if fft_type == "real":
            return np.fft.rfft(self._zeropad_rows(window * block, fftsize))
        elif fft_type == "complex_short":
            return np.fft.fft(self._zeropad_rows(window * block, fftsize))[:, :(1 + (fftsize // 2))]
        elif fft_type == "complex_full":  # For reconstruction
            return np.fft.fft(self._zeropad_rows(window * block, fftsize))
        else:
            raise ValueError("Invalid fft_type {}, must use 'real', 'complex_short', "
                             "or 'complex_full'".format(fft_type))

    @staticmethod
    def _reassignment_value_map(x: np.ndarray, reassignment_mode: str) -> np.ndarray:
        if reassignment_mode == "magsq":
            return np.abs(x) ** 2.0
        elif reassignment_mode == "complex":
            return x
        else:
            raise ValueError("Invalid reassignment_mode {}".format(reassignment_mode))

    def _reassign_sst(self, f: np.ndarray, f_plus: np.ndarray, window: np.ndarray,
                      fftsize: int, sstsize: int, reassignment_mode: str) -> np.ndarray:
        channels = self.AudioSignal.channels
        num_bins_up_to_nyquist = (sstsize // 2) + 1
        sst_shape = (channels, num_bins_up_to_nyquist)  # Bins above Nyquist generally irrelevant for SST purposes

        wft = self.wft(f, window, fftsize)
        wft_plus = self.wft(f_plus, window, fftsize)
        rf = self._calculate_rf(wft, wft_plus)  # Unit: Normalized frequency
        out_of_bounds = np.where((rf < 0) | (rf > 0.5))  # For real valued signals rf > 0.5 is meaningless
        wft[out_of_bounds] = 0
        rf[out_of_bounds] = 0
        sst_out = np.zeros(sst_shape, dtype=(complex if reassignment_mode == "complex" else float))

        for channel in range(channels):
            np.add.at(sst_out[channel], (rf[channel] * sstsize).astype(int),
                      self._reassignment_value_map(wft[channel], reassignment_mode))
        return sst_out

    def _reassign_jtfrt(self, write_frame: int, f: np.ndarray, f_plus: np.ndarray,
                        window: np.ndarray, fftsize: int, sstsize: int, reassignment_mode: str):
        jtfrt_memory_num_frames = self.jtfrt_memory.shape[0]
        jtfrt_memory_num_frames_half = jtfrt_memory_num_frames // 2
        jtfrt_frames_back = jtfrt_memory_num_frames_half - 1
        jtfrt_frames_front = jtfrt_memory_num_frames_half
        channels = self.AudioSignal.channels

        wft = self.wft(f, window, fftsize)
        wft_plus_freq = self.wft(f_plus, window, fftsize)
        # WARNING: For JTFRM based on QSTFT with multiple FFT sizes, OR FFT size changed by user,
        # the line below will need to be modified.
        try:
            f_plus_time = f * self.exp_time_shift[:, :fftsize]  # See warning above
        except IndexError:
            raise IndexError("self.exp_time_shift has dimensions {}, "
                             "but FFT size passed here is {}".format(self.exp_time_shift.shape, fftsize))
        wft_plus_time = self.wft(f_plus_time, window, fftsize, fft_type="complex_short")

        rf = self._calculate_rf(wft, wft_plus_freq)  # Unit: Normalized frequency
        rtdev = self._calculate_rtdev(wft, wft_plus_time)
        rtdev = np.zeros(rtdev.shape, dtype=rtdev.dtype)  # For debug, see if it works like SST does
        out_of_bounds = np.where((rf < 0) | (rf > 0.5) | (rtdev > jtfrt_frames_back) | (rtdev < -jtfrt_frames_front))
        wft[out_of_bounds] = 0
        rf[out_of_bounds] = 0

        # Change rf, rt to the appropriate location indices
        rf = (rf * sstsize).astype(int)
        rt = (np.round(write_frame - rtdev) % jtfrt_memory_num_frames).astype(int)  # % because memory array is circular
        rt[out_of_bounds] = 0

        for channel in range(channels):
            np.add.at(self.jtfrt_memory[:, channel, :], (rt[channel], rf[channel]),
                      self._reassignment_value_map(wft[channel], reassignment_mode))

    def _calculate_rf(self, wft: np.ndarray, wft_plus: np.ndarray) -> np.ndarray:
        eps_division = self.param_dict["eps_division"]
        return np.angle(wft_plus / (wft + eps_division)) / TWOPI  # Unit: Normalized frequency

    def _calculate_rtdev(self, wft: np.ndarray, wft_plus: np.ndarray):
        eps_division = self.param_dict["eps_division"]
        hopsize = self.param_dict["hopsize"]
        windowsize = self.param_dict["windowsize"]
        samplerate = self.AudioSignal.samplerate

        # Returned unit is in STFT frames from the center, current frame.
        return np.angle(wft_plus / (wft + eps_division)) / TWOPI / hopsize * samplerate + windowsize/2/hopsize

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
        inputsize = input_array.shape[1]
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
                output_array = np.pad(input_array, ((0, 0), (padsize_left, padsize_right)), mode='reflect')
            else:
                raise ValueError("input array to pad_boundary_rows has dimensions {}, "
                                 "which is not supported... must be 2D array even if mono".format(input_array.shape))

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
        inputsize = input_array.shape[1]
        if inputsize == finalsize:
            return input_array
        else:
            padsize = finalsize - inputsize
            padsize_left = padsize // 2
            padsize_right = padsize - padsize_left

            if len(input_array.shape) == 2:
                output_array = np.pad(input_array, ((0, 0), (padsize_left, padsize_right)), mode='constant')
            else:
                raise ValueError("input array to zeropad_rows has dimensions {}, "
                                 "which is not supported... must be 2D array even if mono".format(input_array.shape))

            return output_array
