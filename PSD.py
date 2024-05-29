import numpy as np
import matplotlib.pyplot as plt
import requests
import scipy.signal

# Download the file
file_url = 'https://github.com/Duchstf/quench-detector/blob/signal-analysis/sample-data/Ramp21/ai0.npy?raw=true'
local_filename = 'ai0.npy'

# Use requests to download the file
response = requests.get(file_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Load the downloaded .npy file
signal = np.load(local_filename)

# Define the sampling frequency in MHz
Fs_MHz = 1  

# Convert Fs to Hz
Fs_Hz = Fs_MHz * 1_000_000

# Calculate PSD using periodogram
f_periodogram, S_periodogram = scipy.signal.periodogram(signal, Fs_Hz, scaling='density')

# Calculate PSD using Welch's method
f_welch, S_welch = scipy.signal.welch(signal, Fs_Hz, nperseg=1024)

# Plot the PSD calculated by periodogram
plt.figure()
plt.semilogy(f_periodogram, S_periodogram)
plt.xlim([0, 100])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density using Periodogram')
plt.show()
