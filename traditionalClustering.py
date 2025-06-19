from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import time


class TraditionalClustering:
    #data_xy, residuals, labels_true, clusters)
    def getTraditionalClustering(self, data_xy, residuals, labels_true, k):
        results = {}
        predicted_labels = {}
        runtimes = {}

        for method, dataset, name in [
            (GaussianMixture(n_components=k, random_state=19), data_xy, "gmm_xy"),
            (GaussianMixture(n_components=k, random_state=19), residuals.reshape(-1, 1), "gmm_res"),
            (KMeans(n_clusters=k, random_state=19), data_xy, "kmeans_xy"),
            (KMeans(n_clusters=k, random_state=19), residuals.reshape(-1, 1), "kmeans_res"),
            (SpectralClustering(n_clusters=k, random_state=19, affinity='nearest_neighbors'), data_xy, "spectral_xy"), # , affinity='nearest_neighbors'
            (SpectralClustering(n_clusters=k, random_state=19, affinity='nearest_neighbors'), residuals.reshape(-1, 1), #, affinity='nearest_neighbors'
             "spectral_res")
        ]:
            start_time = time.time()
            model = method.fit(dataset)
            labels_pred = model.predict(dataset) if hasattr(model, 'predict') else model.labels_

            # Store predicted labels
            predicted_labels[name] = labels_pred
            end_time = time.time()
            runtimes[name] = end_time - start_time
            # Replace accuracy and F1-score with ARI, NMI, and FMI
            ari = adjusted_rand_score(labels_true, labels_pred)
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            fmi = fowlkes_mallows_score(labels_true, labels_pred)

            results[f"{name}_ARI"] = ari
            results[f"{name}_NMI"] = nmi
            results[f"{name}_FMI"] = fmi

        return results, predicted_labels, runtimes
