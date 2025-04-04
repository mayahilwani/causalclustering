import numpy as np
from datetime import datetime
from top_sort import *
import RFunctions as rf
import matplotlib

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class Plotting:
    def __init__(self, data):
        self.data = data

    def plot_2d_results(self, only_one, pa_i, variable_index, results, y_pred, y_pred_clusters=None, true_labels=None):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]
        k = len(np.unique(results))  # Number of clusters
        colors = get_cmap("tab10", k)  # Generate k distinct colors

        fig, ax = plt.subplots(figsize=(7, 7))

        if not only_one:
            for i in range(k):
                X_group = X[results == i, :]
                y_group = y[results == i]
                ax.scatter(X_group, y_group, color=colors(i), alpha=0.5, label=f'Group {i + 1}')

                if y_pred_clusters is not None:
                    sorted_indices = np.argsort(X_group.flatten())
                    ax.plot(X_group[sorted_indices], y_pred_clusters[i][sorted_indices], linestyle='--', linewidth=2,
                            color=colors(i))
        else:
            ax.scatter(X, y, color='purple', alpha=0.5, label='All Data')

        # Plot the overall MARS line
        sorted_indices_all = np.argsort(X.flatten())
        ax.plot(X[sorted_indices_all], y_pred[sorted_indices_all], color='green', linestyle='-', linewidth=2,
                label='MARS (All Data)')

        ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Second plot for true_labels
        if true_labels is not None:
            fig, ax = plt.subplots(figsize=(7, 7))
            unique_labels = np.unique(true_labels)
            colors_true = get_cmap("tab10", len(unique_labels))

            for i, label in enumerate(unique_labels):
                X_group = X[true_labels == label]
                y_group = y[true_labels == label]
                ax.scatter(X_group, y_group, color=colors_true(i), alpha=0.5, label=f'Label {label}')

            ax.set_title(f'True Labels: Variable {variable_index} vs Parent {pa_i}')
            ax.set_xlabel(f'Parent Variable {pa_i}')
            ax.set_ylabel(f'Variable {variable_index}')
            ax.legend()
            plt.tight_layout()
            plt.show()

    def plot_2d_other(self, pa_i, variable_index, labels, name):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter = ax.scatter(X, y, c=labels, cmap='viridis', alpha=0.5)
        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        plt.show()

    def plot_3d_results(self, only_one, pa_i, variable_index, results, rearth, y_pred_clusters=None):
        X_parent1 = self.data[:, pa_i[0]]
        X_parent2 = self.data[:, pa_i[1]]
        y = self.data[:, variable_index]
        k = len(np.unique(results))
        colors = get_cmap("tab10", k)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if not only_one:
            for i in range(k):
                X_group = self.data[results == i, :][:, pa_i]
                y_group = y[results == i]
                ax.scatter(X_group[:, 0], X_group[:, 1], y_group, color=colors(i), alpha=0.5, label=f'Group {i + 1}')
        else:
            ax.scatter(X_parent1, X_parent2, y, color='purple', alpha=0.5, label='All Data')

        x1_range = np.linspace(X_parent1.min(), X_parent1.max(), 50)
        x2_range = np.linspace(X_parent2.min(), X_parent2.max(), 50)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        X_mesh = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
        y_pred_mesh = rf.predict_mars(X_mesh, rearth).reshape(X1_mesh.shape)

        ax.plot_surface(X1_mesh, X2_mesh, y_pred_mesh, color='green', alpha=0.3)

        ax.set_title(f'Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_3d_other(self, pa_i, variable_index, labels, name):
        X_parent1 = self.data[:, pa_i[0]]
        X_parent2 = self.data[:, pa_i[1]]
        y = self.data[:, variable_index]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_parent1, X_parent2, y, c=labels, cmap='viridis', alpha=0.5)
        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        plt.show()


'''
import numpy as np;
from datetime import datetime
from top_sort import *
import RFunctions as rf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class Plotting:
    def __init__(self, data):
        self.data = data

    def plot_2d_results(self, only_one, pa_i, variable_index, results, y_pred, y_pred1 = None, y_pred2 = None , true_labels = None):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        if not only_one:
            X_group1 = X[results == 0, :]
            y_group1 = y[results == 0]
            X_group2 = X[results == 1, :]
            y_group2 = y[results == 1]
            ax.scatter(X_group1, y_group1, color='blue', alpha=0.5, label='Group 1')
            ax.scatter(X_group2, y_group2, color='red', alpha=0.5, label='Group 2')
            # Plot the MARS line for group1
            sorted_indices_g1 = np.argsort(X_group1.flatten())
            X_g1_sorted = X_group1[sorted_indices_g1]
            y_pred_g1_sorted = y_pred1[sorted_indices_g1]
            ax.plot(X_g1_sorted, y_pred_g1_sorted, color='purple', linestyle='--', linewidth=2,
                    label='MARS (Group1)')
            # Plot the MARS line for group2
            sorted_indices_g2 = np.argsort(X_group2.flatten())
            X_g2_sorted = X_group2[sorted_indices_g2]
            y_pred_g2_sorted = y_pred2[sorted_indices_g2]
            ax.plot(X_g2_sorted, y_pred_g2_sorted, color='orange', linestyle='--', linewidth=2,
                    label='MARS (Group2)')
        # Plot the MARS line for the full dataset
        sorted_indices_all = np.argsort(X.flatten())
        X_all_sorted = X[sorted_indices_all]
        y_all_sorted = y_pred[sorted_indices_all]
        ax.plot(X_all_sorted, y_all_sorted, color='green', linestyle='-', linewidth=2,
                label='MARS (All Data)')
        if only_one:
            ax.scatter(X, y, color='purple', alpha=0.5, label='Group 1')

        ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        X_orig = X[true_labels == 0]
        y_orig = y[true_labels == 0]
        X_intv = X[true_labels == 1]
        y_intv = y[true_labels == 1]
        ax.scatter(X_orig, y_orig, color='blue', alpha=0.5, label='Original')
        ax.scatter(X_intv, y_intv, color='red', alpha=0.5, label='Intervention')
        ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_2d_other(self, pa_i, variable_index, labels, name):
        X = self.data[:,pa_i]
        y = self.data[:,variable_index]
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter = ax.scatter(X, y, c=labels, cmap='viridis', alpha=0.5)
        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        plt.show()

    def plot_3d_results(self, only_one, pa_i, variable_index, results, rearth, y_pred1 = None, y_pred2 = None , true_labels = None):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Extract parent variables
        X_group1 = X[results == 0, :]
        y_group1 = y[results == 0]
        X_group2 = X[results == 1, :]
        y_group2 = y[results == 1]
        X_parent1 = self.data[:, pa_i[0]]
        X_parent2 = self.data[:, pa_i[1]]
        y_all = self.data[:, variable_index]
        if not only_one:
            # Plot Group 1
            ax.scatter(X_group1[:, 0], X_group1[:, 1], y_group1, color='blue', alpha=0.5, label='Group 1')

            # Plot Group 2
            ax.scatter(X_group2[:, 0], X_group2[:, 1], y_group2, color='red', alpha=0.5, label='Group 2')
        else:
            ax.scatter(X_parent1, X_parent2, y_all, color='purple', alpha=0.5, label='Group 1')
        # Create a meshgrid for surface plot (MARS predictions over full dataset)
        x1_range = np.linspace(X_parent1.min(), X_parent1.max(), 50)
        x2_range = np.linspace(X_parent2.min(), X_parent2.max(), 50)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        X_mesh = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
        y_pred_mesh = rf.predict_mars(X_mesh, rearth).reshape(
            X1_mesh.shape)

        # Plot the MARS surface for the full dataset
        ax.plot_surface(X1_mesh, X2_mesh, y_pred_mesh, color='green', alpha=0.3,
                        label='MARS (All Data)')

        # You can add separate surfaces or predictions for Group 1 and Group 2 if needed
        # ax.plot_surface(X1_g1_mesh, X2_g1_mesh, y_pred1_mesh, color='purple', alpha=0.3, label='MARS (Group1)')
        # ax.plot_surface(X1_g2_mesh, X2_g2_mesh, y_pred2_mesh, color='orange', alpha=0.3, label='MARS (Group2)')

        # Set labels and title
        ax.set_title(f'Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_3d_other(self, pa_i, variable_index, labels, name):
        X_parent1 = self.data[:,pa_i[0]]
        X_parent2 = self.data[:,pa_i[1]]
        y = self.data[:,variable_index]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_parent1, X_parent2, y, c=labels, cmap='viridis', alpha=0.5)
        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        plt.show()'''