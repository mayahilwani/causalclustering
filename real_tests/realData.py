from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

def run_code(n, k, r, mdl_th):
    # Filepath
    datapath = f"/Users/mayahilwani/PycharmProjects/msc-mhilwani/real_tests" #f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(datapath, n, k, [8], r, mdl_th )

def main():

    #run_code(1,2,False, False)
    data_file = "full_data.txt"
    data = np.loadtxt(data_file, delimiter=',')
    # Create label arrays for the full dataset
    true_labels_6 = (data[:, 11] == 3).astype(int)
    true_labels_8 = (data[:, 11] == 4).astype(int)
    # Load predicted/found labels from files
    found_labels_6 = np.loadtxt("truth_labels_6.txt", dtype=int).flatten()
    found_labels_8 = np.loadtxt("truth_labels_8.txt", dtype=int).flatten()

    # Compute ARI scores
    ari_6 = adjusted_rand_score(true_labels_6, found_labels_6)
    ari_8 = adjusted_rand_score(true_labels_8, found_labels_8)

    print(f"ARI for node 6 (experiment 3): {ari_6:.4f}")
    print(f"ARI for node 8 (experiment 4): {ari_8:.4f}")

    # Pastel colors
    pastel_blue = "#AEC6CF"
    pastel_orange = "#FFDAB9"
    colors = [pastel_blue, pastel_orange]

    def plot_node_vs_parents(node_index, parent_indices, labels, title, filename):
        node_values = data[:, node_index]
        parent1_values = data[:, parent_indices[0]]
        parent2_values = data[:, parent_indices[1]]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, parent_values, parent_idx in zip(axes, [parent1_values, parent2_values], parent_indices):
            for label in np.unique(labels):
                mask = labels == label
                ax.scatter(
                    parent_values[mask],
                    node_values[mask],
                    label=f"Cluster {label}",
                    alpha=0.6,
                    color=colors[label]
                )
            ax.set_xlabel(f"Parent X{parent_idx}")
            ax.set_ylabel(f"Node X{node_index}")
            ax.set_title(f"X{node_index} vs X{parent_idx}")

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {filename}")

    '''# Plot for node 6 (parents 4, 7)
    plot_node_vs_parents(
        node_index=6,
        parent_indices=[4, 7],
        labels=found_labels_6,
        title=f"Node 6 vs Parents (Found Labels)\nARI: {ari_6:.2f}",
        filename="node6_found.pdf"
    )

    # Plot for node 8 (parents 3, 2)
    plot_node_vs_parents(
        node_index=8,
        parent_indices=[3, 2],
        labels=found_labels_8,
        title=f"Node 8 vs Parents (Found Labels)\nARI: {ari_8:.2f}",
        filename="node8_found.pdf"
    )'''

    plot_node_vs_parents(
        node_index=6,
        parent_indices=[4, 7],
        labels=true_labels_6,
        title=f"Node 6 vs Parents (True Labels)\nARI: {ari_6:.2f}",
        filename="node6_true.pdf"
    )

    # Plot for node 8 (parents 3, 2)
    plot_node_vs_parents(
        node_index=8,
        parent_indices=[3, 2],
        labels=true_labels_8,
        title=f"Node 8 vs Parents (True Labels)\nARI: {ari_8:.2f}",
        filename="node8_true.pdf"
    )
    '''# === CLUSTERING EVALUATION ===

    print(f"\n--- ARI Scores from Clustering Methods ---")

    # --- Experiment 3: Node 6 (variables 6, 4, 7) ---
    X_6 = data[:, [6, 4, 7]]

    # GMM
    gmm_6 = GaussianMixture(n_components=2, random_state=42).fit(X_6)
    gmm_labels_6 = gmm_6.predict(X_6)
    print(f"GMM ARI (Node 6): {adjusted_rand_score(true_labels_6, gmm_labels_6):.4f}")

    # KMeans
    kmeans_6 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_6)
    kmeans_labels_6 = kmeans_6.labels_
    print(f"KMeans ARI (Node 6): {adjusted_rand_score(true_labels_6, kmeans_labels_6):.4f}")

    # Spectral Clustering
    spectral_6 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit(X_6)
    spectral_labels_6 = spectral_6.labels_
    print(f"Spectral ARI (Node 6): {adjusted_rand_score(true_labels_6, spectral_labels_6):.4f}")

        # --- Experiment 4: Node 8 (variables 8, 3, 2) ---
        X_8 = data[:, [8, 3, 2]]

    # GMM
    gmm_8 = GaussianMixture(n_components=2, random_state=42).fit(X_8)
    gmm_labels_8 = gmm_8.predict(X_8)
    print(f"\nGMM ARI (Node 8): {adjusted_rand_score(true_labels_8, gmm_labels_8):.4f}")

    # KMeans
    kmeans_8 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_8)
    kmeans_labels_8 = kmeans_8.labels_
    print(f"KMeans ARI (Node 8): {adjusted_rand_score(true_labels_8, kmeans_labels_8):.4f}")

    # Spectral Clustering
    spectral_8 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit(X_8)
    spectral_labels_8 = spectral_8.labels_
    print(f"Spectral ARI (Node 8): {adjusted_rand_score(true_labels_8, spectral_labels_8):.4f}")
'''

'''
    # (6, 4, 7, 3)
    # Select intervention samples (experiment 3)
    intv_data_6 = data[data[:, 11] == 3]

    # Select the rest (non-intervention on experiment 3)
    non_intv_data_6 = data[data[:, 11] != 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter for observational data
    ax.scatter(non_intv_data_6[:, 6], non_intv_data_6[:, 4], non_intv_data_6[:, 7],
               alpha=0.2, label="Observational data", color="gray")

    # 3D scatter for interventional data
    ax.scatter(intv_data_6[:, 6], intv_data_6[:, 4], intv_data_6[:, 7],
               alpha=0.6, label="Interventional data", color="blue")

    # Labels and title
    ax.set_title('Variable 6 vs Parent Variables 4 and 7 (3D view)')
    ax.set_xlabel('Variable 6')
    ax.set_ylabel('Parent Variable 4')
    ax.set_zlabel('Parent Variable 7')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # (8, 3, 2, 4)
    # Select intervention samples (experiment 4)
    intv_data_8 = data[data[:, 11] == 4]

    # Select the rest (non-intervention on experiment 4)
    non_intv_data_8 = data[data[:, 11] != 4]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter for observational data
    ax.scatter(non_intv_data_8[:, 8], non_intv_data_8[:, 3], non_intv_data_8[:, 2],
               alpha=0.2, label="Observational data", color="gray")

    # 3D scatter for interventional data
    ax.scatter(intv_data_8[:, 8], intv_data_8[:, 3], intv_data_8[:, 2],
               alpha=0.6, label="Interventional data", color="blue")

    # Labels and title
    ax.set_title('Variable 8 vs Parent Variables 3 and 2 (3D view)')
    ax.set_xlabel('Variable 8')
    ax.set_ylabel('Parent Variable 3')
    ax.set_zlabel('Parent Variable 2')
    ax.legend()
    plt.tight_layout()
    plt.show()'''


if __name__ == "__main__":
    main()