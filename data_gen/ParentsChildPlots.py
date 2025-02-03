from top_sort import *
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def plot_data(gt, data):
    dims = gt.shape[1]

    g = Graph(dims)
    for i in range(dims):
        for j in range(dims):
            if gt[i, j] == 1:
                g.addEdge(i, j);

    ordering = g.nonRecursiveTopologicalSort()
    for variable_index in ordering:
        # Find parent indices for the current variable
        pa_i = np.where(gt[:, variable_index] == 1)[0]
        print(f"Variable {variable_index} has parents {pa_i}")
        # Number of parents
        num_parents = len(pa_i)

        # If there are no parents, we skip plotting for this variable
        if num_parents == 0:
            print(f"Variable {variable_index} has no parents.")
            continue

        # Set up the plotting window
        fig, axes = plt.subplots(nrows=1, ncols=num_parents, figsize=(5 * num_parents, 5))
        if num_parents == 1:
            axes = [axes]  # Ensure axes is iterable

        # Define indices for the two sections of the data
        first_section = np.arange(5000)  # First 5000 data points
        last_section = np.arange(len(data) - 1000, len(data))  # Last 1000 data points

        # Plot the relationship between the variable and each of its parents
        for idx, parent_index in enumerate(pa_i):
            ax = axes[idx]
            # Plot the first 5000 points in one color
            ax.scatter(data[first_section, parent_index], data[first_section, variable_index], 
                       color='blue', alpha=0.5, label="First 5000 points")

            # Plot the last 1000 points in another color
            ax.scatter(data[last_section, parent_index], data[last_section, variable_index], 
                       color='red', alpha=0.5, label="Last 1000 points")

            ax.set_title(f'Variable {variable_index} vs Parent {parent_index}')
            ax.set_xlabel(f'Parent {parent_index}')
            ax.set_ylabel(f'Variable {variable_index}')
            ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

def main():
    try:
        base_path = "./maya_data/experimentNEWflip1"
        gt_file = f"{base_path}/truth1.txt"
        gt = np.loadtxt(gt_file, delimiter=',')
        data_file1 = f"{base_path}/data1.txt"
        data_file2 = f"{base_path}/dataintv1.txt"
        data1 = np.loadtxt(data_file1, delimiter=',')
        print('DATA1: ' + str(data1.shape))
        data2 = np.loadtxt(data_file2, delimiter=',')
        print('DATA2: ' + str(data2.shape))
        if data1.shape[1] != data2.shape[1]:
            raise ValueError("The two files must have the same number of columns for vertical concatenation.")
        # Merge the data by concatenating vertically
        data = np.vstack((data1, data2))
        print(data.shape)
        plot_data(gt, data)

    except Exception as e:(
        print(f"An error occurred: {e}"))


if __name__ == "__main__":
    main()