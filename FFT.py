import numpy as np
import matplotlib.pyplot as plt
import requests

# Download the file
file_url = 'https://github.com/Duchstf/quench-detector/blob/signal-analysis/sample-data/Ramp21/ai0.npy?raw=true'
local_filename = 'ai0.npy'

# Use requests to download the file
response = requests.get(file_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Load the downloaded .npy file
signal = np.load(local_filename)

# Define the sampling frequency
Fs = 100_000  # 100 kHz

# Calculate the Fourier Transform of the signal
fft_result = np.fft.fft(signal)
n = len(signal)
f = np.fft.fftfreq(n, 1/Fs)

# Compute the magnitude of the FFT result
magnitude = np.abs(fft_result)

# Only take the positive part of the frequency spectrum
f_positive = f[:n//2]
magnitude_positive = magnitude[:n//2]

# Plot the magnitude spectrum
plt.plot(f_positive, magnitude_positive)
plt.xlim([0, 50000])  # Limiting to Nyquist frequency (Fs/2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude Spectrum')
plt.show()
