'''
This file is for testing purposes only.
'''
from top_sort import *
import numpy as np
import matplotlib.pyplot as plt
import RFunctions as rf
import seaborn as sns
from sklearn.mixture import GaussianMixture

def fit_mars(X, y):
    """ Fits a MARS model and returns the residuals and model. """
    sse, coeffs, no_of_terms, interactions, rearth = rf.REarth(X, y, M=1)
    #print(f'MARS RETURN: SSE={sse}, Coeffs={coeffs}, Terms={no_of_terms}, Interactions={interactions}, Model={rearth}')
    # Predict and calculate residuals
    y_pred = rf.predict_mars(X, rearth)
    residuals = y - y_pred

    return residuals, rearth, y_pred

def plot_residuals(gt, data):
    dims = gt.shape[1]
    g = Graph(dims)
    for i in range(dims):
        for j in range(dims):
            if gt[i, j] == 1:
                g.addEdge(i, j)

    ordering = g.nonRecursiveTopologicalSort()
    for variable_index in ordering:
        pa_i = np.where(gt[:, variable_index] == 1)[0]
        print(f"Variable {variable_index} has parents {pa_i}")

        if len(pa_i) == 0:
            print(f"Variable {variable_index} has no parents.")
            continue

        X = data[:, pa_i]
        y = data[:, variable_index]
        print('X SHAPE is: ' + str(X.shape))

        # Fit MARS on the entire dataset
        residuals, rearth, y_pred = fit_mars(X, y)

        '''# Sort X and y_pred for continuous plotting
        sorted_indices = np.argsort(X.flatten())
        X_sorted = X[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]'''

        '''# Identify large residuals
        mean_residual = np.mean(np.abs(residuals))
        std_residual = np.std(residuals)
        margin = mean_residual + 1.0 * std_residual
        large_residuals = np.abs(residuals) > margin
        normal_residuals = ~large_residuals'''

        # Fit a Gaussian Mixture Model with 2 components
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data.reshape(-1, 1))

        # Predict the cluster for each point
        labels = gmm.predict(data.reshape(-1, 1))

        # Split the data into two groups based on the labels
        group1 = data[labels == 0]
        group2 = data[labels == 1]

        # Optional: Visualize the split
        import matplotlib.pyplot as plt

        plt.hist(group1, bins=30, alpha=0.5, label='Group 1', density=True)
        plt.hist(group2, bins=30, alpha=0.5, label='Group 2', density=True)
        plt.plot(x_values, kde_values, label='KDE', color='black')
        plt.legend()
        plt.show()

        # Plot residuals KDE for the full dataset
        sns.kdeplot(residuals, shade=True)
        '''fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ax.scatter(np.arange(len(residuals))[large_residuals], residuals[large_residuals], color='red', alpha=0.7,
                   label='Large Residuals')
        ax.scatter(np.arange(len(residuals))[normal_residuals], residuals[normal_residuals], color='blue', alpha=0.5,
                   label='Normal Residuals')
        ax.set_title(f'Variable {variable_index} Residuals')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Residual Value')
        ax.legend()

        plt.tight_layout()
        plt.show()'''
        if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
            ax.scatter(X, y, color='blue', alpha=0.5, label='All Data')
            ax.scatter(X[large_residuals], y[large_residuals], color='red', alpha=0.7, label='High Residuals')

            # Plot the MARS line for the full dataset
            ax.plot(X, y_pred, color='green', linestyle='-', linewidth=2, label='MARS (All Data)')
            # TODO

            # If large residuals exist, refit MARS on them and plot the line
            if np.any(large_residuals):
                X_large = X[large_residuals].reshape(-1, 1)
                y_large = y[large_residuals]
                if len(X_large) > 0:
                    _, rearth_large, y_pred_large = fit_mars(X_large, y_large)
                    # Sort again for plotting
                    sorted_indices_large = np.argsort(X_large.flatten())
                    X_large_sorted = X_large[sorted_indices_large]
                    y_pred_large_sorted = y_pred_large[sorted_indices_large]

                    ax.plot(X_large_sorted, y_pred_large_sorted, color='purple', linestyle='--', linewidth=2,
                            label='MARS (High Residuals)')

            # Fit and plot the MARS line for the normal residuals
            X_normal = X[normal_residuals].reshape(-1, 1)
            y_normal = y[normal_residuals]
            if len(X_normal) > 0:
                _, rearth_normal, y_pred_normal = fit_mars(X_normal, y_normal)
                # Sort again for plotting
                sorted_indices_normal = np.argsort(X_normal.flatten())
                X_normal_sorted = X_normal[sorted_indices_normal]
                y_pred_normal_sorted = y_pred_normal[sorted_indices_normal]

                ax.plot(X_normal_sorted, y_pred_normal_sorted, color='orange', linestyle='-.', linewidth=2,
                        label='MARS (Normal Residuals)')

            #if len(pa_i) > 1:
             #   # TODO

            ax.set_title(f'Variable {variable_index} vs Parent {pa_i[0]}')
            ax.set_xlabel(f'Parent Variable {pa_i[0]}')
            ax.set_ylabel(f'Variable {variable_index}')
            ax.legend()

            plt.tight_layout()
            plt.show()



def main():
    try:
        base_path = "C:/Users/ziadh/Documents/CausalGen-Osman/maya_data/experimentNEWshift3"
        gt_file = f"{base_path}/truth1.txt"
        gt = np.loadtxt(gt_file, delimiter=',')
        data_file1 = f"{base_path}/data1.txt"
        data_file2 = f"{base_path}/dataintv1.txt"
        data1 = np.loadtxt(data_file1, delimiter=',')
        data2 = np.loadtxt(data_file2, delimiter=',')
        if data1.shape[1] != data2.shape[1]:
            raise ValueError("The two files must have the same number of columns for vertical concatenation.")
        data = np.vstack((data1, data2))
        print(data.shape)
        plot_residuals(gt, data)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
