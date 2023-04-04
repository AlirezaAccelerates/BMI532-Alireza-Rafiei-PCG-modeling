# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Create KDE plots for each dataset (Data1, Data2, Data3) with a shaded area and specified alpha
sns.kdeplot(Data1, label='Data1', linewidth=2, shade=True, alpha=.15)
sns.kdeplot(Data2, label='Data2', linewidth=2, shade=True ,alpha=.15)
sns.kdeplot(Data1, label='Data3', linewidth=2, shade=True ,alpha=.15)

# Set the labels for the x and y axis
plt.xlabel('Data Values')
plt.ylabel('Probability Density')

# Add a legend to the plot
plt.legend()

# Add a title to the plot
plt.title('Comparison of Data1, Data2, and Data3 Distributions')

# Save the plot as an image file with specified dpi (dots per inch) value
plt.savefig("image.jpg", dpi = 600)
