import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.signal import welch

# Download the file
file_url = 'https://github.com/Duchstf/quench-detector/blob/signal-analysis/sample-data/Ramp21/ai0.npy?raw=true'
local_filename = 'ai0.npy'

response = requests.get(file_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Define the analysis functions
def plot_fft(signal, sample_rate, max_freq=1000):
    signal -= np.mean(signal)
    fft_result = np.fft.fft(signal)
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/sample_rate)
    magnitude_spectrum = np.abs(fft_result)
    
    dc_component = np.abs(fft_result[0])
    normalized_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    
    mask = (freq > 0) & (freq <= max_freq)

    plt.figure(figsize=(10, 6))
    plt.plot(freq[mask], normalized_spectrum[mask])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    plt.title('FFT Analysis')
    plt.show(block=False)

def plot_time_domain(signal, sample_rate):
    time = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title('Time Domain Signal')
    plt.show(block=False)

def plot_psd(signal, sample_rate):
    freqs, psd = welch(signal, sample_rate)
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.grid(True)
    plt.title('Power Spectral Density')
    plt.show(block=False)

# Load the data
signal = np.load(local_filename)
sample_rate = 1000  # Adjust this to the correct sample rate for your data

# Call the analysis functions
plot_time_domain(signal, sample_rate)
plot_fft(signal, sample_rate)
plot_psd(signal, sample_rate)

# Keep the plots open
plt.show()
