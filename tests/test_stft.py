from adaptft.tft import TFTransformer
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/home/rejinal/Datasets/SAFE/VJRS_05_062.08.wav"

TFT = TFTransformer(audio_path)
stft = np.asarray([i for i in TFT.compute_stft()])
scaled_stft = np.log(1 + np.abs(stft) ** 2.0)

# Get rid of the channel axis
scaled_stft = np.reshape(scaled_stft, [scaled_stft.shape[0], scaled_stft.shape[2]])

plt.pcolormesh(scaled_stft.T, cmap="Greys")
plt.show()
input("Press ENTER to finish")