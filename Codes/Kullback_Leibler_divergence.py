# Required libraries
import numpy as np

def kl_divergence(P, Q):
    P = np.clip(P, 1e-10, 1 - 1e-10)
    Q = np.clip(Q, 1e-10, 1 - 1e-10)
    return np.sum(P * np.log(P / Q))

def data_to_distribution(data, num_bins):
    hist, _ = np.histogram(data, bins=num_bins, density=True)
    return hist / np.sum(hist)

num_bins = 50

# Convert datasets to probability distributions using the data_to_distribution function
P = data_to_distribution(Data1, num_bins)
Q1 = data_to_distribution(Data2, num_bins)
Q2 = data_to_distribution(Data3, num_bins)

# Calculate KL divergence between Data1 and Data2, and Data1 and Data3
kl1 = kl_divergence(P, Q1)
kl2 = kl_divergence(P, Q2)

print("KL Divergence (Data1 and Data2):", kl1)
print("KL Divergence (Data1 and Data3):", kl2)
