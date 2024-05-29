import requests
import numpy as np
import matplotlib.pyplot as plt

class DataDownloader:
    def __init__(self, url):
        self.url = url
        self.file_path = 'ai1.npy'

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
    def __init__(self, data):
        self.data = data

    def perform_fft(self):
        self.fft_result = np.fft.fft(self.data)
        self.fft_magnitude = np.abs(self.fft_result)
        self.fft_frequency = np.fft.fftfreq(len(self.data))
        return self.fft_frequency, self.fft_magnitude


class Plotter:
    def __init__(self, original_data, smoothed_data, fft_frequency, fft_magnitude):
        self.original_data = original_data
        self.smoothed_data = smoothed_data
        self.fft_frequency = fft_frequency
        self.fft_magnitude = fft_magnitude

    def plot(self):
        plt.figure(figsize=(12, 12), constrained_layout=True)

        plt.subplot(3, 1, 1)
        plt.plot(self.original_data, label='Original Data')
        plt.title('Original Signal Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.smoothed_data, label='Preprocessed Data (Smoothed)', color='orange')
        plt.title('Preprocessed Signal Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Amplitude')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.fft_frequency, self.fft_magnitude, label='FFT of Preprocessed Data', color='green')
        plt.title('FFT of Preprocessed Signal Data')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.subplots_adjust(hspace=0.8) 
        plt.show()


if __name__ == "__main__":

    url = 'https://github.com/Duchstf/quench-detector/raw/signal-analysis/sample-data/Ramp20/ai1.npy'
    downloader = DataDownloader(url)
    file_path = downloader.download()

    preprocessor = DataPreprocessor(file_path)
    data_normalized = preprocessor.normalize()
    data_smoothed = preprocessor.smooth()

    
    fft_processor = FFTProcessor(data_smoothed)
    fft_frequency, fft_magnitude = fft_processor.perform_fft()

   
    plotter = Plotter(preprocessor.data, data_smoothed, fft_frequency, fft_magnitude)
    plotter.plot()
