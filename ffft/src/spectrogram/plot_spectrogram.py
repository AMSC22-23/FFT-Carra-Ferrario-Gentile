import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the first line to get the dimensions
        rows, cols = map(int, file.readline().strip().split())

        # Read the rest of the lines into a 2D list
        matrix = [list(map(float, line.strip().split())) for line in file]

    return np.array(matrix), rows, cols

def plot_all_txt_files(directory):
    # Use glob to match all .txt files in the directory and sort them
    txt_files = sorted(glob.glob(os.path.join(directory, '*.txt')))
    num_files = len(txt_files)

    # Calculate the number of rows and columns for the subplot grid
    num_cols = int(np.ceil(np.sqrt(num_files)))
    num_rows = int(np.ceil(num_files / num_cols))

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for ax, file_path in zip(axes, txt_files):
        matrix, rows, cols = read_matrix_from_file(file_path)
        im = ax.imshow(matrix, cmap='gray')
        ax.set_title(os.path.basename(file_path))
        plt.colorbar(im, ax=ax)

    # Hide any unused subplots
    for ax in axes[num_files:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("test.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_spectrogram.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("The provided path is not a directory.")
        sys.exit(1)
    # Plot the matrices from all .txt files in the specified directory in order
    plot_all_txt_files(directory)