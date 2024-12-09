import numpy as np
import matplotlib.pyplot as plt

# Load the histogram data from CSV
histogram_data = np.loadtxt("hog_histograms.csv", delimiter=",")

# Aggregate the data by summing over all cells
aggregated_histogram = np.sum(histogram_data, axis=0)

# Create a simple histogram plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(aggregated_histogram)), aggregated_histogram, color='skyblue', edgecolor='black')
plt.title("Aggregated HOG Histogram", fontsize=14)
plt.xlabel("Orientation Bin", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
plt.xticks(range(len(aggregated_histogram)))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("hog_histogram_plot.png")
plt.show()

