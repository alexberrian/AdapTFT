from adaptft.tft import TFTransformer
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/home/rejinal/Datasets/SAFE/VJRS_05_062.08.wav"

TFT = TFTransformer(audio_path)
jtfrt = np.asarray([i for i in TFT.compute_jtfrt()])
scaled_jtfrt = np.log(1 + jtfrt ** 2.0)

# Get rid of the channel axis
scaled_jtfrt = np.reshape(scaled_jtfrt, [scaled_jtfrt.shape[0], scaled_jtfrt.shape[2]])

plt.pcolormesh(scaled_jtfrt.T, cmap="Greys")
plt.show()
input("Press ENTER to finish")