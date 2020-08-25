from soundfile import SoundFile
import os

"""
For audio i/o; essentially just does a filename check then gets SoundFile as a whole.
"""


class AudioSignal(SoundFile):

    def __init__(self, filename):
        self.filename = filename
        self.filename_check()
        super(SoundFile, self).__init__(filename)

    def filename_check(self):
        if self.filename is None:
            raise IOError("Must specify filename")
        elif not os.path.exists(self.filename):
            raise IOError("File {} does not exist".format(self.filename))
        elif not os.path.isfile(self.filename):
            raise IOError("Filename {} is not a valid file (is it a folder?)".format(self.filename))

