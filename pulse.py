import numpy as np


def spectrum_from_time(envelope):
    return len(envelope) * np.fft.ifft(envelope, axis=0)


def time_from_spectrum(spectrum):
    return np.fft.fft(spectrum, axis=0) / len(spectrum)


def synthesize_at_time(spectrum, frequencies, time, start_time):
    weights = np.exp(-1j * frequencies * (time - start_time)) / len(frequencies)
    return np.einsum('f,f...->...', weights, spectrum)
