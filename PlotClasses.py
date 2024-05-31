import requests
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

class DataDownloader:
    def __init__(self, url):
        self.url = url
        self.file_path = 'ai0.npy'

    def download(self):
        response = requests.get(self.url)
        with open(self.file_path, 'wb') as f:
            f.write(response.content)
        return self.file_path

class DataPreprocessor:
    def __init__(self, file_path):
        self.data = np.load(file_path, allow_pickle=True)

    def normalize(self):
        self.data_normalized = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self.data_normalized

    def smooth(self, window_size=5):
        self.data_smoothed = np.convolve(self.data_normalized, np.ones(window_size)/window_size, mode='valid')
        return self.data_smoothed

class FFTProcessor:
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate

    def perform_fft(self):
        # Calculate the Fourier Transform of the signal
        fft_result = np.fft.fft(self.data)
        n = len(self.data)
        f = np.fft.fftfreq(n, 1/self.sampling_rate)

        # Compute the magnitude of the FFT result
        magnitude = np.abs(fft_result)

        # Only take the positive part of the frequency spectrum
        f_positive = f[:n//2]
        magnitude_positive = magnitude[:n//2]

        return f_positive, magnitude_positive

class PSDProcessor:
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate

    def periodogram(self):
        f_periodogram, S_periodogram = scipy.signal.periodogram(self.data, self.sampling_rate, scaling='density')
        return f_periodogram, S_periodogram

    def welch(self, nperseg=1024):
        f_welch, S_welch = scipy.signal.welch(self.data, self.sampling_rate, nperseg=nperseg)
        return f_welch, S_welch

class Plotter:
    def __init__(self, original_data, smoothed_data, fft_frequency, fft_magnitude, f_periodogram, S_periodogram, f_welch, S_welch):
        self.original_data = original_data
        self.smoothed_data = smoothed_data
        self.fft_frequency = fft_frequency
        self.fft_magnitude = fft_magnitude
        self.f_periodogram = f_periodogram
        self.S_periodogram = S_periodogram
        self.f_welch = f_welch
        self.S_welch = S_welch

    def plot(self):
        plt.figure(figsize=(12, 18), constrained_layout=True)

        plt.subplot(4, 1, 1)
        plt.plot(self.original_data, label='Original Data')
        plt.title('Original Signal Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(self.smoothed_data, label='Preprocessed Data (Smoothed)', color='orange')
        plt.title('Preprocessed Signal Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Amplitude')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(self.fft_frequency, self.fft_magnitude, label='FFT of Preprocessed Data', color='green')
        plt.title('FFT Magnitude Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.semilogy(self.f_periodogram, self.S_periodogram, label='PSD using Periodogram', color='blue')
        plt.semilogy(self.f_welch, self.S_welch, label='PSD using Welch', color='red')
        plt.xlim([0, max(self.f_periodogram)])  # Limiting to Nyquist frequency (Fs/2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Power Spectral Density')
        plt.legend()

        plt.subplots_adjust(hspace=0.8) 
        plt.show()

if __name__ == "__main__":
    url = 'https://github.com/Duchstf/quench-detector/raw/signal-analysis/sample-data/Ramp21/ai0.npy'
    downloader = DataDownloader(url)
    file_path = downloader.download()

    preprocessor = DataPreprocessor(file_path)
    data_normalized = preprocessor.normalize()
    data_smoothed = preprocessor.smooth()

    # Set the sampling rate to 100 kHz
    sampling_rate = 100_000

    fft_processor = FFTProcessor(data_smoothed, sampling_rate)
    fft_frequency, fft_magnitude = fft_processor.perform_fft()

    psd_processor = PSDProcessor(data_smoothed, sampling_rate)
    f_periodogram, S_periodogram = psd_processor.periodogram()
    f_welch, S_welch = psd_processor.welch()

    plotter = Plotter(preprocessor.data, data_smoothed, fft_frequency, fft_magnitude, f_periodogram, S_periodogram, f_welch, S_welch)
    plotter.plot()
