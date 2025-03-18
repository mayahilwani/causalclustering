from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
import numpy as np
from scipy.optimize import linear_sum_assignment


class TraditionalClustering:
    def compute_f1(self, labels_pred, labels_true):
        return f1_score(labels_true, labels_pred, average='weighted')


    def match_labels_sklearn(self, labels_pred, labels_true):
        unique_pred = np.unique(labels_pred)  # Unique predicted labels
        unique_true = np.unique(labels_true)  # Unique true labels

        k_pred = len(unique_pred)
        k_true = len(unique_true)

        # Create a contingency matrix
        contingency = np.zeros((k_pred, k_true))

        for i, pred_label in enumerate(unique_pred):
            for j, true_label in enumerate(unique_true):
                contingency[i, j] = np.sum((labels_pred == pred_label) & (labels_true == true_label))

        # Use Hungarian algorithm for optimal label assignment
        row_ind, col_ind = linear_sum_assignment(-contingency)

        # Mapping from predicted labels to true labels
        label_mapping = {unique_pred[row]: unique_true[col] for row, col in zip(row_ind, col_ind)}

        # Apply the mapping to the predicted labels
        return np.vectorize(label_mapping.get)(labels_pred)

    def match_labels(self, labels_pred, labels_true, k):
        contingency = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                contingency[i, j] = np.sum((labels_pred == i) & (labels_true == j))
        row_ind, col_ind = linear_sum_assignment(-contingency)
        label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
        return np.vectorize(label_mapping.get)(labels_pred)
    '''
    def match_labels(labels_pred, labels_true):
    unique_pred = np.unique(labels_pred)  # Unique labels in predicted
    unique_true = np.unique(labels_true)  # Unique labels in true
    
    k_pred = len(unique_pred)  # Number of predicted clusters
    k_true = len(unique_true)  # Number of true clusters
    
    # Create contingency matrix with shape (k_pred, k_true)
    contingency = np.zeros((k_pred, k_true))
    
    # Fill contingency matrix
    for i, pred_label in enumerate(unique_pred):
        for j, true_label in enumerate(unique_true):
            contingency[i, j] = np.sum((labels_pred == pred_label) & (labels_true == true_label))
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-contingency)  # Maximize agreement
    
    # Create label mapping
    label_mapping = {unique_pred[row]: unique_true[col] for row, col in zip(row_ind, col_ind)}
    
    # Map labels_pred to the new matched labels
    return np.vectorize(label_mapping.get)(labels_pred)
    '''
    def getTraditionalClustering(self, data_xy, residuals, labels_true, k):
        results = {}

        for method, dataset, name in [(GaussianMixture(n_components=k, random_state=19), data_xy, "gmm_xy"),
                                      (GaussianMixture(n_components=k, random_state=19), residuals.reshape(-1, 1),
                                       "gmm_res"),
                                      (KMeans(n_clusters=k, random_state=19), data_xy, "kmeans_xy"),
                                      (KMeans(n_clusters=k, random_state=19), residuals.reshape(-1, 1), "kmeans_res"),
                                      (SpectralClustering(n_clusters=k, random_state=19, affinity='nearest_neighbors'),
                                       data_xy, "spectral_xy"),
                                      (SpectralClustering(n_clusters=k, random_state=19, affinity='nearest_neighbors'),
                                       residuals.reshape(-1, 1), "spectral_res")]:
            model = method.fit(dataset)
            labels_pred = model.predict(dataset) if hasattr(model, 'predict') else model.labels_
            labels_pred = self.match_labels(labels_pred, labels_true, k)

            accuracy = np.mean(labels_pred == labels_true)
            f1 = self.compute_f1(labels_pred, labels_true)

            results[f"{name}_accuracy"] = accuracy
            results[f"{name}_f1"] = f1

        return results