# Import required libraries
import numpy as np

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

# Create histograms for each dataset (Data1, Data2, Data3) with 50 bins and normalize them
hist_data1, _ = np.histogram(Data1, bins=50, density=True)
hist_data2, _ = np.histogram(Data2, bins=50, density=True)
hist_data3, _ = np.histogram(Data3, bins=50, density=True)

# Calculate the Total Variation Distance between the histograms of Data1 and Data2
tvd1 = total_variation_distance(hist_data1, hist_data2)
print("Total Variation Distance (Data1 and Data2):", tvd1)

# Calculate the Total Variation Distance between the histograms of Data1 and Data3
tvd2 = total_variation_distance(hist_data1, hist_data3)
print("Total Variation Distance (Data1 and Data3):", tvd2)
