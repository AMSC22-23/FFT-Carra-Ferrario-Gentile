import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Check if a directory path is provided
if len(sys.argv) != 2:
    print("Usage: python plot_spectrograms.py <path_to_directory>")
    sys.exit(1)

# Get the directory path from the command line argument
directory_path = sys.argv[1]

# Function to plot a single spectrogram
def plot_spectrogram(data, nrows, ncols, filename):
    # Create a new figure for the spectrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    # Compute aspect ratio
    aspect_ratio = ncols / (2 * nrows)
    # Plot the spectrogram
    cax = ax.imshow(data, aspect=aspect_ratio, cmap='magma', vmin=0, vmax=30, origin='lower')
    fig.colorbar(cax, ax=ax)
    # Save the figure in a specified directory
    fig.savefig(os.path.join(directory_path, filename), bbox_inches='tight')
    # Close the figure to prevent overlap with next plots
    plt.close(fig)

# Walk through the directory and read each file
for idx, filename in enumerate(os.listdir(directory_path)):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r') as f:
            # Read the first line to get the number of rows and columns
            nrows, ncols = map(int, f.readline().split())
            
            # Read the spectrogram data
            data = np.array([list(map(float, line.split())) for line in f])
        
        # Generate a filename for the PNG
        png_filename = filename.replace('.txt', '.png')
        # Plot and save the spectrogram as a PNG
        plot_spectrogram(data, nrows, ncols, png_filename)
