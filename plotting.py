import numpy as np
from top_sort import *
import RFunctions as rf
import matplotlib
from datetime import datetime
import os


matplotlib.use('TkAgg')  # REMOVE IT LATER

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

# Apply visual style
sns.set_style("whitegrid")
plt.rcParams.update({
    'lines.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

class Plotting:
    def __init__(self, data):
        self.data = data
        self.pastel_colors = [
            '#7BAFD4', # blue
            '#66C2A5',  # muted green
            '#FC8D62',  # soft orange
            '#E5C494',  # sand
            '#8DA0CB',  # light blue-violet
            '#E78AC3',  # lavender pink
            '#FFD92F',  # warm yellow
            '#A6D854',  # green-yellow
            '#B3B3B3'  # grey
        ]

    def plot_2d_results(self, only_one, pa_i, variable_index, results, y_pred, y_pred_clusters=None, true_labels=None):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]
        k = len(np.unique(results))

        fig, ax = plt.subplots(figsize=(7, 7))

        if not only_one:
            for i in range(k):
                X_group = X[results == i, :]
                y_group = y[results == i]
                color = self.pastel_colors[i % len(self.pastel_colors)]
                ax.scatter(X_group, y_group, color=color, alpha=0.6, label=f'Group {i + 1}')

                if y_pred_clusters is not None:
                    sorted_indices = np.argsort(X_group.flatten())
                    ax.plot(X_group[sorted_indices], y_pred_clusters[i][sorted_indices],
                            linestyle='--', linewidth=2, color=color)
        else:
            ax.scatter(X, y, color='#A8DADC', alpha=0.6, label='All Data')

        sorted_indices_all = np.argsort(X.flatten())
        ax.plot(X[sorted_indices_all], y_pred[sorted_indices_all], color='#118AB2', linestyle='-', linewidth=2,
                label='MARS (All Data)')

        ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()

        # Optional save
        #plt.savefig(f'plot_variable_{variable_index}_parent_{pa_i}.pdf', format='pdf', dpi=300)
        '''
        # Set custom save directory
        save_dir = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/final_results/saved_plots"
        os.makedirs(save_dir, exist_ok=True)  # Create if it doesn't exist

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_variable_{variable_index}_parent_{pa_i}_{timestamp}.pdf"

        # Full save path
        save_path = os.path.join(save_dir, filename)

        # Save the figure
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Plot saved to: {save_path}")
        #'''

        plt.show()

        if true_labels is not None:
            fig, ax = plt.subplots(figsize=(7, 7))
            unique_labels = np.unique(true_labels)

            for i, label in enumerate(unique_labels):
                X_group = X[true_labels == label]
                y_group = y[true_labels == label]
                color = self.pastel_colors[i % len(self.pastel_colors)]
                ax.scatter(X_group, y_group, color=color, alpha=0.6, label=f'Label {label}')

            ax.set_title(f'True Labels: Variable {variable_index} vs Parent {pa_i}')
            ax.set_xlabel(f'Parent Variable {pa_i}')
            ax.set_ylabel(f'Variable {variable_index}')
            ax.legend()
            plt.tight_layout()

            # Optional save
            # plt.savefig(f'true_labels_variable_{variable_index}_parent_{pa_i}.pdf', format='pdf', dpi=300)


            plt.show()

    def plot_2d_other(self, pa_i, variable_index, labels, name):
        X = self.data[:, pa_i]
        y = self.data[:, variable_index]
        unique_labels = np.unique(labels)

        fig, ax = plt.subplots(figsize=(7, 7))

        for i, label in enumerate(unique_labels):
            color = self.pastel_colors[i % len(self.pastel_colors)]
            ax.scatter(X[labels == label], y[labels == label], color=color, alpha=0.6, label=f'Cluster {label}')

        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parent {pa_i}')
        ax.set_xlabel(f'Parent Variable {pa_i}')
        ax.set_ylabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()

        # Optional save
        #plt.savefig(f'{name}_2d_clusters_variable_{variable_index}_parent_{pa_i}.pdf', format='pdf', dpi=300)
        '''
        # Set custom save directory
        save_dir = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/final_results/saved_plots"
        os.makedirs(save_dir, exist_ok=True)  # Create if it doesn't exist

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_variable_{variable_index}_parent_{pa_i}_{timestamp}.pdf"

        # Full save path
        save_path = os.path.join(save_dir, filename)

        # Save the figure
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Plot saved to: {save_path}")
        #'''

        plt.show()

    def plot_3d_results(self, only_one, pa_i, variable_index, results, rearth, y_pred_clusters=None):
        X_parent1 = self.data[:, pa_i[0]]
        X_parent2 = self.data[:, pa_i[1]]
        y = self.data[:, variable_index]
        k = len(np.unique(results))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if not only_one:
            for i in range(k):
                X_group = self.data[results == i, :][:, pa_i]
                y_group = y[results == i]
                color = self.pastel_colors[i % len(self.pastel_colors)]
                ax.scatter(X_group[:, 0], X_group[:, 1], y_group, color=color, alpha=0.6, label=f'Group {i + 1}')
        else:
            ax.scatter(X_parent1, X_parent2, y, color='#A8DADC', alpha=0.6, label='All Data')

        x1_range = np.linspace(X_parent1.min(), X_parent1.max(), 50)
        x2_range = np.linspace(X_parent2.min(), X_parent2.max(), 50)
        X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
        X_mesh = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
        y_pred_mesh = rf.predict_mars(X_mesh, rearth).reshape(X1_mesh.shape)

        ax.plot_surface(X1_mesh, X2_mesh, y_pred_mesh, color='#118AB2', alpha=0.3)

        ax.set_title(f'Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()

        # Optional save
        # plt.savefig(f'plot_3d_variable_{variable_index}_parents_{pa_i[0]}_{pa_i[1]}.pdf', format='pdf', dpi=300)

        plt.show()

    def plot_3d_other(self, pa_i, variable_index, labels, name):
        X_parent1 = self.data[:, pa_i[0]]
        X_parent2 = self.data[:, pa_i[1]]
        y = self.data[:, variable_index]
        unique_labels = np.unique(labels)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(unique_labels):
            color = self.pastel_colors[i % len(self.pastel_colors)]
            mask = labels == label
            ax.scatter(X_parent1[mask], X_parent2[mask], y[mask], color=color, alpha=0.6, label=f'Cluster {label}')

        ax.set_title(f'{name} Clusters for Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
        ax.set_xlabel(f'Parent Variable {pa_i[0]}')
        ax.set_ylabel(f'Parent Variable {pa_i[1]}')
        ax.set_zlabel(f'Variable {variable_index}')
        ax.legend()
        plt.tight_layout()

        # Optional save
        # plt.savefig(f'{name}_3d_clusters_variable_{variable_index}_parents_{pa_i[0]}_{pa_i[1]}.pdf', format='pdf', dpi=300)

        plt.show()
