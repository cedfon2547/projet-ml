import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

def bandpass_filter(ppg: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to a PPG signal.

    This function applies a Butterworth bandpass filter to the input photoplethysmogram (PPG) signal.
    Bandpass filtering is useful for removing noise and extracting the frequency components of interest
    from the PPG signal, which typically lies within a specific frequency range.

    Notes:
    - The Nyquist frequency is half of the sampling frequency.
    - The Butterworth filter is designed to have a flat frequency response in the passband.
    - Bandpass filtering is particularly useful for PPG signals to isolate the heart rate frequency range,
      typically between 0.5 Hz and 5 Hz, and to remove low-frequency baseline wander and high-frequency noise.

    Parameters:
    ppg (np.ndarray): The input PPG signal as a 1D numpy array.
    lowcut (float): The lower cutoff frequency of the bandpass filter in Hz.
    highcut (float): The upper cutoff frequency of the bandpass filter in Hz.
    fs (float): The sampling frequency of the PPG signal in Hz.
    order (int, optional): The order of the Butterworth filter. Default is 5.

    Returns:
    np.ndarray: The filtered PPG signal as a 1D numpy array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, ppg)

def get_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Identify the peaks in a photoplethysmogram (PPG) signal.

    This function is useful for analyzing PPG data to detect heartbeats. 
    Peaks in the PPG signal correspond to the systolic peaks of the cardiac cycle, 
    which can be used to calculate heart rate and other cardiovascular metrics.

    Parameters:
    ppg (np.ndarray): The PPG signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    np.ndarray: Indices of the peaks in the PPG signal.
    """
    return signal.find_peaks(ppg, distance=fs/2)[0]

def peak_to_peak_interval(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the average peak-to-peak interval of a PPG signal.

    This function is useful for analyzing PPG data because the peak-to-peak interval
    can provide insights into the heart rate and its variability. By calculating the
    average time between peaks, we can estimate the average heart rate and detect
    any irregularities in the heart rhythm.

    Parameters:
    ppg (np.ndarray): The photoplethysmogram (PPG) signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    float: The average peak-to-peak interval in seconds.
    """
    peaks, _ = signal.find_peaks(ppg)
    intervals = np.diff(peaks) / fs
    return np.mean(intervals)

def pulse_rate_variability(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the pulse rate variability (PRV) from a photoplethysmogram (PPG) signal.

    The pulse rate variability (PRV) is a measure of the variation in time between consecutive heartbeats.
    It is derived from the PPG signal, which is a non-invasive method to monitor cardiovascular activity.
    PRV is useful for assessing autonomic nervous system function and can provide insights into cardiovascular health,
    stress levels, and overall well-being.

    Parameters:
    ppg (np.ndarray): The PPG signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    float: The standard deviation of the intervals between detected peaks in the PPG signal.
    """
    peaks, _ = signal.find_peaks(ppg)
    intervals = np.diff(peaks) / fs
    return np.std(intervals)

def heart_rate(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the heart rate from a photoplethysmogram (PPG) signal.

    This function is useful for analyzing PPG data to determine the heart rate,
    which is a critical vital sign. The PPG signal is often used in wearable
    health devices to monitor cardiovascular health, as it provides a non-invasive
    way to measure the heart rate by detecting blood volume changes in the microvascular
    bed of tissue.

    Parameters:
    ppg (np.ndarray): The PPG signal as a numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    float: The heart rate in beats per minute (BPM).
    """
    ppi = peak_to_peak_interval(ppg, fs)
    return 60.0 / ppi

def statistical_features(ppg: np.ndarray) -> dict:
    """
    Calculate statistical features from a PPG (Photoplethysmogram) signal.

    These statistical features are useful for analyzing PPG data as they provide insights 
    into the central tendency, variability, and shape of the signal distribution. 
    This information can be used for various applications such as detecting anomalies, 
    classifying different physiological states, and improving the accuracy of health monitoring systems.

    Parameters:
    ppg (np.ndarray): A numpy array containing the PPG signal data.

    Returns:
    dict: A dictionary containing the following statistical features:
        - 'mean': The mean value of the PPG signal.
        - 'median': The median value of the PPG signal.
        - 'std': The standard deviation of the PPG signal.
        - 'skewness': The skewness of the PPG signal.
        - 'kurtosis': The kurtosis of the PPG signal.
    """
    return {
        'mean': np.mean(ppg),
        'median': np.median(ppg),
        'std': np.std(ppg),
        'skewness': skew(ppg),
        'kurtosis': kurtosis(ppg)
    }

def rms(ppg: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) of a given PPG (photoplethysmogram) signal.

    The RMS value is a statistical measure of the magnitude of a varying quantity. 
    It is especially useful in the context of PPG data as it provides a measure of 
    the signal's power and can help in identifying the overall energy of the PPG signal. 
    This can be useful for various analyses, such as assessing the quality of the signal 
    or detecting abnormalities in the cardiovascular system.

    Parameters:
    ppg (np.ndarray): A numpy array containing the PPG signal data.

    Returns:
    float: The RMS value of the PPG signal.
    """
    return np.sqrt(np.mean(np.square(ppg)))

def zero_crossing_rate(ppg: np.ndarray) -> float:
    """
    Calculate the zero crossing rate of a PPG signal.

    The zero crossing rate is the rate at which the signal changes sign. 
    This can be useful in analyzing PPG (Photoplethysmogram) data as it 
    provides information about the frequency characteristics of the signal, 
    which can be related to heart rate and other physiological parameters.

    Parameters:
    ppg (np.ndarray): The PPG signal as a numpy array.

    Returns:
    float: The zero crossing rate of the PPG signal.
    """
    zero_crossings = np.where(np.diff(np.sign(ppg)))[0]
    return len(zero_crossings) / len(ppg)

def signal_energy(ppg: np.ndarray) -> float:
    """
    Calculate the energy of a given PPG (photoplethysmogram) signal.

    The energy of a signal is defined as the sum of the squares of its amplitude values.
    This function computes the energy of the PPG signal, which can be useful for various
    signal processing and analysis tasks, such as detecting the presence of certain
    physiological events or assessing the overall signal quality.

    Parameters:
    ppg (np.ndarray): A numpy array containing the PPG signal data.

    Returns:
    float: The energy of the PPG signal.
    """
    return np.sum(np.square(ppg))

def power_spectral_density(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculate the Power Spectral Density (PSD) of a photoplethysmogram (PPG) signal.

    The Power Spectral Density (PSD) is useful for analyzing the frequency content of the PPG signal.
    By examining the PSD, one can identify the dominant frequencies and their power, which can be 
    indicative of physiological parameters such as heart rate variability and respiratory rate.
    This analysis is crucial for understanding the underlying cardiovascular and respiratory dynamics 
    in the PPG signal.

    Parameters:
    ppg (np.ndarray): The PPG signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    np.ndarray: Frequencies corresponding to the PSD values.
    np.ndarray: Power Spectral Density values of the PPG signal.
    """
    f, Pxx = signal.welch(ppg, fs)
    return f, Pxx

def spectral_entropy(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the spectral entropy of a photoplethysmogram (PPG) signal.

    Spectral entropy is a measure of the complexity or randomness of a signal's power spectral density (PSD).
    It quantifies the distribution of power across different frequency components of the signal.

    Spectral entropy is useful in analyzing PPG data because it provides insights into the variability and 
    irregularity of the cardiovascular system. Higher spectral entropy indicates a more complex and less 
    predictable signal, which can be associated with various physiological conditions. It can be used in 
    applications such as detecting arrhythmias, assessing autonomic nervous system activity, and monitoring 
    overall cardiovascular health.

    Parameters:
    ppg (np.ndarray): The PPG signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    float: The spectral entropy of the PPG signal.
    """
    f, Pxx = power_spectral_density(ppg, fs)
    Pxx_norm = Pxx / np.sum(Pxx)
    return -np.sum(Pxx_norm * np.log2(Pxx_norm))

def spectral_energy(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the spectral energy of a PPG signal.

    The spectral energy is a measure of the total power within the signal's frequency spectrum.
    When applied to PPG data, it can provide insights into the overall power distribution of the 
    signal, which is useful for analyzing the cardiovascular system and detecting abnormalities 
    in heart rate variability and other physiological parameters.

    Parameters:
    ppg (np.ndarray): The photoplethysmogram (PPG) signal as a numpy array.
    fs (float): The sampling frequency of the PPG signal.

    Returns:
    float: The spectral energy of the PPG signal.
    """
    _, Pxx = power_spectral_density(ppg, fs)
    return np.sum(Pxx)

def spectral_centroid(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the spectral centroid of a PPG signal.

    The spectral centroid is a measure used in digital signal processing to 
    characterize a spectrum. It indicates where the center of mass of the 
    spectrum is located. In the context of PPG (Photoplethysmogram) signals, 
    the spectral centroid can provide insights into the distribution of 
    frequencies within the signal, which can be useful for analyzing the 
    cardiovascular system and detecting anomalies.

    Parameters:
    ppg (np.ndarray): The PPG signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal.

    Returns:
    float: The spectral centroid of the PPG signal.
    """
    f, Pxx = power_spectral_density(ppg, fs)
    return np.sum(f * Pxx) / np.sum(Pxx)

def peak_frequency(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the peak frequency of a photoplethysmogram (PPG) signal.

    This function computes the power spectral density (PSD) of the given PPG signal
    and returns the frequency at which the PSD is maximized. The peak frequency is
    a useful feature in PPG analysis as it can provide insights into the dominant
    frequency components of the signal, which are often related to physiological
    parameters such as heart rate.

    Parameters:
    ppg (np.ndarray): The input PPG signal as a 1D NumPy array.
    fs (float): The sampling frequency of the PPG signal in Hz.

    Returns:
    float: The frequency at which the power spectral density is maximized.
    """
    f, Pxx = power_spectral_density(ppg, fs)
    return f[np.argmax(Pxx)]

def bandpower(ppg: np.ndarray, fs: float, lowcut: float, highcut: float) -> float:
    """
    Calculate the band power of a given PPG signal within a specified frequency range.

    This function is useful for analyzing PPG data because it allows for the extraction of specific frequency components
    that are of interest. For example, different frequency bands in PPG signals can be associated with various physiological
    phenomena such as heart rate, respiration, and other cardiovascular activities. By calculating the band power, one can
    quantify the energy present in these frequency bands, which can be used for further analysis and interpretation of
    cardiovascular health and other related metrics.

    Parameters:
    ppg (np.ndarray): The photoplethysmogram (PPG) signal as a 1D numpy array.
    fs (float): The sampling frequency of the PPG signal in Hz.
    lowcut (float): The lower bound of the frequency range in Hz.
    highcut (float): The upper bound of the frequency range in Hz.

    Returns:
    float: The power of the PPG signal within the specified frequency band.
    """
    f, Pxx = power_spectral_density(ppg, fs)
    band = np.logical_and(f >= lowcut, f <= highcut)
    return np.sum(Pxx[band])

def heart_rate_variability(ppg: np.ndarray, fs: float) -> float:
    """
    Calculate the heart rate variability (HRV) from photoplethysmogram (PPG) data.

    Heart rate variability is a measure of the variation in time between each heartbeat.
    It is an important indicator of autonomic nervous system function and cardiovascular health.
    This function uses power spectral density (PSD) analysis to estimate HRV within a specific frequency band.

    Notes:
    - The frequency band used for HRV calculation is between 0.6 Hz and 2.5 Hz.
    - This function assumes that the input PPG signal is preprocessed and free of noise.

    Parameters:
    ppg (np.ndarray): The PPG signal as a numpy array.
    fs (float): The sampling frequency of the PPG signal.

    Returns:
    float: The heart rate variability value calculated from the PPG signal.
    """
    f, Pxx = power_spectral_density(ppg, fs)
    band = np.logical_and(f >= 0.6, f <= 2.5)
    return np.sum(Pxx[band])

def ppg_features(ppg: np.ndarray, fs: float) -> dict:
    return {
        **statistical_features(ppg),
        'heart_rate': heart_rate(ppg, fs),
        'peak_to_peak_interval': peak_to_peak_interval(ppg, fs),
        'pulse_rate_variability': pulse_rate_variability(ppg, fs),
        'rms': rms(ppg),
        'zero_crossing_rate': zero_crossing_rate(ppg),
        'signal_energy': signal_energy(ppg),
        'spectral_entropy': spectral_entropy(ppg, fs),
        'spectral_energy': spectral_energy(ppg, fs),
        'spectral_centroid': spectral_centroid(ppg, fs),
        'peak_frequency': peak_frequency(ppg, fs),
        'bandpower': bandpower(ppg, fs, 0.5, 10),
        'heart_rate_variability': heart_rate_variability(ppg, fs)
    }