# Required libraries
import numpy as np
from scipy.spatial.distance import euclidean

# Hellinger Distance function
def hellinger_distance(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

# Create histograms for each dataset (Data1, Data2, Data3) with 50 bins and normalize them
hist_Data1, _ = np.histogram(Data1, bins=50, density=True)
hist_Data2, _ = np.histogram(Data2, bins=50, density=True)
hist_Data3, _ = np.histogram(Data3, bins=50, density=True)

# Calculate the Hellinger Distance between the histograms of Data1 and Data2
hellinger_dist = hellinger_distance(hist_Data1, hist_Data2)
print("Hellinger Distance (Data1 and Data2):", hellinger_dist)

# Calculate the Hellinger Distance between the histograms of Data1 and Data3
hellinger_dist2 = hellinger_distance(hist_Data1, hist_Data3)
print("Hellinger Distance (Data1 and Data3):", hellinger_dist2)
