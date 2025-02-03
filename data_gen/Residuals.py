from top_sort import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#from ipywidgets import interact, IntSlider
#from pyearth import Earth  # Importing the Earth model

def plot_data(gt, data):
    dims = gt.shape[1]

    g = Graph(dims)
    for i in range(dims):
        for j in range(dims):
            if gt[i, j] == 1:
                g.addEdge(i, j)

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

        # Plot the relationship between the variable and each of its parents
        for idx, parent_index in enumerate(pa_i):
            ax = axes[idx]
            # Scatter plot of data points
            ax.scatter(data[:, parent_index], data[:, variable_index], alpha=0.5, label='Data Points')

            '''
            # Fit a MARS model (didn't work) DID POLY TEMP
            X = data[:, parent_index].reshape(-1, 1)
            y = data[:, variable_index]
            model = Earth() ##
            model.fit(X, y) ##
            '''
            # Fitting Polynomial Regression to the dataset
            X = data[:, parent_index].reshape(-1, 1)
            y = data[:, variable_index]
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            poly.fit(X_poly, y)
            lin2 = LinearRegression()
            lin2.fit(X_poly, y)

            # Calculate residuals
            #y_pred = model.predict(X)
            y_pred = lin2.predict(X_poly)
            residuals = y - y_pred
            '''
            threshold = np.percentile(np.abs(residuals), 95)  # Define a threshold for large residuals
            large_residuals = np.abs(residuals) > threshold
            '''
            # Margin using mean and standard deviation, include most points (e.g., 2 standard deviations)
            mean_residual = np.mean(np.abs(residuals))
            std_residual = np.std(residuals)
            margin = mean_residual + 2 * std_residual
            large_residuals = np.abs(residuals) > margin

            # Highlight points with large residuals
            ax.scatter(X[large_residuals], y[large_residuals], color='red', alpha=0.7, label='Large Residuals')

            # Plot the MARS regression line
            sorted_indices = np.argsort(X, axis=0).flatten()
            ax.plot(X[sorted_indices], y_pred[sorted_indices], color='orange', linewidth=2, label='MARS Regression Line')

            ax.set_title(f'Variable {variable_index} vs Parent {parent_index}')
            ax.set_xlabel(f'Parent {parent_index}')
            ax.set_ylabel(f'Variable {variable_index}')
            ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

def main():
    try:
        base_path = "./mintv_data/experimentM2"
        gt_file = f"{base_path}/truth1.txt"
        gt = np.loadtxt(gt_file, delimiter=',')
        data_file1 = f"{base_path}/data1.txt"
        data_file2 = f"{base_path}/dataintv1.txt"
        data1 = np.loadtxt(data_file1, delimiter=',')
        data2 = np.loadtxt(data_file2, delimiter=',')
        if data1.shape[1] != data2.shape[1]:
            raise ValueError("The two files must have the same number of columns for vertical concatenation.")
        # Merge the data by concatenating vertically
        data = np.vstack((data1, data2))
        print(data.shape)
        plot_data(gt, data)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
