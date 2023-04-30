from scipy.signal import resample
from sklearn.preprocessing import normalize

# Resample every signal to 512 Hz
resampled_data = np.empty((len(data_sec), 64*3))  # create an empty array to hold the resampled data
for i in range(len(data_sec)):
    # Normalize the signal to have unit norm
    signal = normalize(data_sec[i].reshape(-1, 1))
    signal = signal.flatten()
    
    # Resample the signal to 512 Hz
    resampled_signal = resample(signal, num=64*3)
    
    # Store the resampled signal in the resampled_data array
    resampled_data[i] = resampled_signal
