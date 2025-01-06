'''
This is just for testing purposes
'''
from top_sort import *
import numpy as np
import matplotlib.pyplot as plt
import RFunctions as rf
import seaborn as sns
from sklearn.mixture import GaussianMixture
from diptest import diptest


def fit_mars(X, y):
    """ Fits a MARS model and returns the residuals and model. """
    sse, coeffs, no_of_terms, interactions, rearth = rf.REarth(X, y, M=1)
    # print(f'MARS RETURN: SSE={sse}, Coeffs={coeffs}, Terms={no_of_terms}, Interactions={interactions}, Model={rearth}')
    # Predict and calculate residuals
    y_pred = rf.predict_mars(X, rearth)

    return rearth, y_pred


def plot_residuals(gt, data):
    # Get topological order of nodes
    dims = gt.shape[1]
    g = Graph(dims)
    for i in range(dims):
        for j in range(dims):
            if gt[i, j] == 1:
                g.addEdge(i, j)
    ordering = g.nonRecursiveTopologicalSort()

    # Traverse the nodes top to bottom
    for variable_index in ordering:
        # Get parents of the node
        pa_i = np.where(gt[:, variable_index] == 1)[0]
        print(f"Variable {variable_index} has parents {pa_i}")

        # If node has no parents print and skip it
        if len(pa_i) == 0:
            print(f"Variable {variable_index} has no parents.")
            continue

        X = data[:, pa_i]
        y = data[:, variable_index]
        print('X SHAPE is: ' + str(X.shape))
        print('Y SHAPE is: ' + str(y.shape))

        # Fit MARS on the entire dataset
        rearth, y_pred = fit_mars(X, y)
        residuals = y - y_pred

        # Perform Hartigan's Dip Test on the residuals
        dip_stat, p_value = diptest(residuals)
        print(f'Dip test p-value for variable {variable_index}: {p_value}')

        if p_value < 0.2:  # MAYBE HIGHER P_VALUE ???
            # Fit a Gaussian Mixture Model with 2 components
            gmm = GaussianMixture(n_components=2)
            gmm.fit(residuals.reshape(-1, 1))

            # Predict the cluster for each point
            labels = gmm.predict(residuals.reshape(-1, 1))

            # Split the data into two groups based on the labels
            group1 = residuals[labels == 0]
            group2 = residuals[labels == 1]
            last_groups = labels.copy()
            first_iter = True

            while(True):
                # Fit different MARS lines for each group and check residuals
                groups = labels.reshape(-1, 1)
                print('Groups SHAPE is: ' + str(groups.shape))
                X_group1 = X[labels == 0, :]
                y_group1 = y[labels == 0]
                X_group2 = X[labels == 1, :]
                y_group2 = y[labels == 1]

                # GROUP1 MARS
                rearth_group1, y_pred_group1 = fit_mars(X_group1, y_group1)
                # Group2 MARS
                rearth_group2, y_pred_group2 = fit_mars(X_group2, y_group2)

                residuals_group1 = y_group1 - y_pred_group1
                residuals_group2 = y_group2 - y_pred_group2

                # Plot residuals KDE for the full dataset
                sns.kdeplot(residuals, fill=True)
                plt.show()

                # Create subplots for side-by-side plots
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Plot the GMM clustering result
                axes[0].hist(group1, bins=30, alpha=0.5, label='Group 1', density=True)
                axes[0].hist(group2, bins=30, alpha=0.5, label='Group 2', density=True)
                axes[0].legend()
                axes[0].set_title('GMM Clustering Result for node ' + str(variable_index))
                axes[0].set_xlabel('Data values')
                axes[0].set_ylabel('Density')

                # Plot the KDE of the new residuals for each group
                sns.kdeplot(residuals_group1.flatten(), fill=True, ax=axes[1], label='Group 1 Residuals')
                sns.kdeplot(residuals_group2.flatten(), fill=True, ax=axes[1], label='Group 2 Residuals')
                axes[1].legend()
                axes[1].set_title('KDE of New Residuals for Each Group')
                axes[1].set_xlabel('Residual values')
                axes[1].set_ylabel('Density')

                # Display the plots
                plt.tight_layout()
                plt.show()

                if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)

                    '''fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
                    # Plot the MARS line for the full dataset
                    X_all = data[:, pa_i]
                    y_all = data[:, variable_index]
                    ax.scatter(X_all, y_all, color='blue', alpha=0.5, label='All')
                    rearth_all, y_pred_all = fit_mars(X_all, y_all)
                    print('X_all SHAPE is: ' + str(X_all.shape))
                    print('y_pred_all SHAPE is: ' + str(y_pred_all.shape))
                    sorted_indices_all = np.argsort(X_all.flatten())
                    X_all_sorted = X_all[sorted_indices_all]
                    y_all_sorted = y_pred_all[sorted_indices_all]
                    ax.plot(X_all_sorted, y_all_sorted, color='green', linestyle='-', linewidth=2,
                            label='MARS (All Data)')
                    plt.show()
                    '''
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
                    ax.scatter(X_group1, y_group1, color='blue', alpha=0.5, label='Group 1')
                    ax.scatter(X_group2, y_group2, color='red', alpha=0.5, label='Group 2')
                    # Plot the MARS line for the full dataset
                    X_all = data[:, pa_i]
                    y_all = data[:, variable_index]
                    rearth_all, y_pred_all = fit_mars(X_all, y_all)
                    print('X_all SHAPE is: ' + str(X_all.shape))
                    print('y_pred_all SHAPE is: ' + str(y_pred_all.shape))
                    sorted_indices_all = np.argsort(X_all.flatten())
                    X_all_sorted = X_all[sorted_indices_all]
                    y_all_sorted = y_pred_all[sorted_indices_all]
                    ax.plot(X_all_sorted, y_all_sorted, color='green', linestyle='-', linewidth=2,
                            label='MARS (All Data)')

                    # Plot the MARS line for group1
                    sorted_indices_g1 = np.argsort(X_group1.flatten())
                    X_g1_sorted = X_group1[sorted_indices_g1]
                    y_pred_g1_sorted = y_pred_group1[sorted_indices_g1]
                    ax.plot(X_g1_sorted, y_pred_g1_sorted, color='purple', linestyle='--', linewidth=2,
                            label='MARS (Group1)')

                    # Plot the MARS line for group2
                    sorted_indices_g2 = np.argsort(X_group2.flatten())
                    X_g2_sorted = X_group2[sorted_indices_g2]
                    y_pred_g2_sorted = y_pred_group2[sorted_indices_g2]
                    ax.plot(X_g2_sorted, y_pred_g2_sorted, color='orange', linestyle='--', linewidth=2,
                            label='MARS (Group2)')

                    ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

                # Reassign the data points
                for i in range(X.shape[0]):
                    x = X[i, :]
                    y_predict_1 = rf.predict_mars(x, rearth_group1)
                    y_predict_2 = rf.predict_mars(x, rearth_group2)
                    res1 = abs(y[i] - y_predict_1)
                    res2 = abs(y[i] - y_predict_2)
                    if res1 < res2:
                        labels[i] = 0
                    else:
                        labels[i] = 1

                '''if not first_iter and np.array_equal(labels, last_groups):
                    print('Classification is settled!')
                    break'''
                change_threshold = 0.01  # e.g., require less than 1% of changes to stop
                changes = np.sum(labels != last_groups)
                if not first_iter and (changes / len(labels)) < change_threshold:
                    print(f'Classification is settled with {changes} changes!')
                    break

                if first_iter:
                    first_iter = False

                last_groups = labels.copy()


def main():
    try:
        base_path = "C:/Users/ziadh/Documents/CausalGen-Osman/maya_data/experimentNEWflip1"
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
