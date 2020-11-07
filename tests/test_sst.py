from adaptft.tft import TFTransformer
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/home/rejinal/Datasets/SAFE/VJRS_05_062.08.wav"

TFT = TFTransformer(audio_path)
sst = np.asarray([i for i in TFT.compute_sst()])
scaled_sst = np.log(1 + sst ** 2.0)

# Get rid of the channel axis
scaled_sst = np.reshape(scaled_sst, [scaled_sst.shape[0], scaled_sst.shape[2]])

plt.pcolormesh(scaled_sst.T, cmap="Greys")
plt.show()
input("Press ENTER to finish")