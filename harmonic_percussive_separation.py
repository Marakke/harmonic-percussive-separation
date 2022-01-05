# import libraries
from presets import Preset
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import time
import sounddevice as sd

y, sr = lb.load('Muumit goes war.mp3', sr=None, duration=100)
audioLength = lb.get_duration(y, sr=sr)
'''
Usually librosa forces the code to use sampling rate of 22050. 
We'd prefer to use the sampling rate of audio signal loaded.
Therefore we use presets library to change the default sampling rate to 
that of the audio signal. Also in the process, we set the frame length
and frame shift to the values presented in the article.
'''
librosa = Preset(lb)
librosa['sr'] = sr
librosa['n_fft'] = 1024
librosa['hop_length'] = 512

F = lb.stft(y)  # Step 1: Calculate the STFT of the input signal
gamma = 0.3  # Defines the amount of range compression

# Step 2: Calculate a range-compressed version of Power Spectrogram
W = np.power(np.abs(F), 2 * gamma)

# Step 3: Set the initial values for harmonic and percussive spectrogram
H = 0.5 * W
P = 0.5 * W

# Two more values are required in the algorithm, and they are as follows
kMax = 100  # The amount of iterations in the algorithm
alpha = 0.3  # Balance parameter (experimentally determined)

for k in range(kMax):

    # Step 4: Calculate the update variables delta(k)

    # The delta(K) is roughly the difference between harmonic
    # and percussive parts. Slots for those parts are made beforehand.
    partH = np.zeros_like(H)  # An array to store the harmonic part
    partP = np.zeros_like(P)  # An array to store the percussive part

    # The harmonic part is calculated with for loop over the i.
    for iIter in range(1, np.shape(H)[1] - 1):
        partH[:, iIter] = alpha * ((H[:, iIter - 1] -
                                    (2 * H[:, iIter]) +
                                    H[:, iIter + 1]) / 4)

    # The percussive part is calculated with for loop over the h.
    for hIter in range(1, np.shape(H)[0] - 1):
        partP[hIter, :] = (1 - alpha) * ((P[hIter - 1, :] -
                                          (2 * P[hIter, :]) +
                                          P[hIter + 1, :]) / 4)

    deltak = partH - partP  # The calculated update variables

    # Step 5: Update H and P as defined in the article
    H = np.minimum(np.maximum(H + deltak, 0), W)
    P = W - H

    # Step 6: Increment k by 1 ( if k != kmax -1)
    # If we had used while-loop, there would be k+1 here, but with for-loop
    # the incrementation is done correctly automatically (requiring that
    # we used the correct length for the for-loop).

# Step 7: Binarize the separation result
H = np.where(np.less(H, P), 0, W)
P = np.where(np.greater_equal(H, P), 0, W)

# Step 8: Convert H and P into waveforms

#  The H and P are assigned this value in this step because it's needed to
#  plot the spectrogram later. Otherwise, we would calculate the waveforms
#  directly, not with an additional step. We do not care about overwriting
#  existing data on H or P, because they're not needed anymore.
H = np.power(H, (1 / (2 * gamma))) * np.exp(
        1j * np.angle(F))  # ISTFT is taken first on this, with H
P = np.power(P, (1 / (2 * gamma))) * np.exp(
        1j * np.angle(F))  # ISTFT is taken second on this, with P

#  Calculate the actual waveforms with ISTFT. Length is set to len(y)
#  so we can subtract the separated waveforms from the original.
h = lb.istft(H, length=len(y))
p = lb.istft(P, length=len(y))

rp = np.max(np.abs(F))  # To scale the colorbar correctly (hopefully)
plt.figure(figsize=(12, 8))

# Plot the original audio's spectrogram.
plt.subplot(3, 1, 1)
lb.display.specshow(lb.amplitude_to_db(np.abs(F), ref=rp), sr=sr,
                    y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Full spectrogram')
plt.tight_layout()

# Plot the harmonic spectrogram.
plt.subplot(3, 1, 2)
lb.display.specshow(lb.amplitude_to_db(np.abs(H), ref=rp), sr=sr,
                    y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Harmonic spectrogram')
plt.tight_layout()

# Plot the percussive spectrogram.
plt.subplot(3, 1, 3)
lb.display.specshow(lb.amplitude_to_db(np.abs(P), ref=rp), sr=sr,
                    y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Percussive spectrogram')
plt.tight_layout()

plt.show()

e = y - p - h  # Noise (original signal minus percussive & harmonic components)

# Calculate the sinal noise ratio.
SNR = 10*np.log10(np.sum(np.power(y, 2))/np.sum(np.power(e, 2)))
print(SNR)

# Play the original, percussive and harmonic sounds.
sd.play(y, sr)
#time.sleep(audioLength)
sd.play(p, sr)
#time.sleep(audioLength)
sd.play(h, sr)
