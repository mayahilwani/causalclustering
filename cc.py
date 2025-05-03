from node import Node;
from edge import Edge;
from slope import Slope;
from utils import *
from logger import Logger
import numpy as np;
from datetime import datetime
from top_sort import *
import RFunctions as rf
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
from combinator import Combinator
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
import math
from plotting import Plotting
from traditionalClustering import TraditionalClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import SpectralClustering

class CC:
    def __init__(self, max_int, log_results=True, vrb=True, dims=0):
        self.M = max_int
        self.log_flag = log_results
        self.verbose = vrb
        self.V = dims
        self.slope_ = Slope()
        self.plot = None
        self.vars = None #np.zeros((5, 5)); # basically all data
        self.gt = None #np.zeros((5, 5)); # ground truth
        self.attributes = [] # number_of_nodes , num_orig_data, ....
        self.log_path = "./logs/log_" + str(datetime.now(tz=None)).replace(' ', '_') + ".txt"
        self.filename = ""
        self.node_labels = []
        self.split = [] #np.zeros(10);
        self.k = []
        self.score_all = [] #np.zeros(10);  # score full
        self.score_split = [] #np.zeros(10);  # score split
        self.Nodes = []
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9
        self.ordering = []
        self.foundIntv = []
        self.intv = []
        self.Edges = None
        if self.log_flag:
            print("Saving results to: ", self.log_path)

    def loadData(self, filename):
        self.filename = filename
        try:
            # Load GroundTruth, Data, Intventions
            gt_file = f"{self.filename}/truth1.txt"
            data_file = f"{self.filename}/data1.txt"
            intv_file = f"{self.filename}/interventions1.txt"
            self.gt = np.loadtxt(gt_file, delimiter=',')
            data = np.loadtxt(data_file, delimiter=',')
            print("Any NaNs in data?", np.isnan(data).any())
            if np.isnan(data).any():
                nan_rows = np.where(np.isnan(data).any(axis=1))[0]
                print("Rows with NaNs:", nan_rows)
                print("Values in those rows:\n", data[nan_rows])
            try:
                intvs = np.loadtxt(intv_file, delimiter=',', dtype=int)
            except ValueError as e:
                intvs = []  # Or handle as needed
            self.intv = intvs
            self.vars = data
            self.plot = Plotting(self.vars)
            # Standardize the loaded data in self.vars
            #normalized_vars = self.vars  # Standardize(self.vars)
            self.V = self.vars.shape[1]
            self.result = np.zeros((self.V, self.vars.shape[0]))
            self.node_labels = np.zeros((self.V, self.vars.shape[0]))
            self.split = np.zeros(self.V)
            self.k = np.zeros(self.V)
            self.score_all = np.zeros(self.V)
            self.score_split = np.zeros(self.V);
            #self.Nodes = np.zeros(self.V);
            #self.ordering = []
            #self.foundIntv = []
            #self.intv = []
            # Load Attributes
            attributes_file = f"{self.filename}/attributes1.txt"
            with open(attributes_file, "r") as atts:
                lines = atts.readlines()
                values = lines[1].strip()  # Second line contains the values
                # Convert the values to a list
                self.attributes = values.split(", ")
        except Exception as e:
            print(f"An error occurred: {e}")

    def run(self, k = 2, needed_nodes = [], random = False, mdl_th = False):
        if not needed_nodes:
            nodestats_file = f"{self.filename}/node_STATS.txt"
            if random:
                nodestats_file = f"{self.filename}/node_STATS_rand.txt"
            with open(nodestats_file, "w") as stats:
                stats.write("id,num_parents,k,true_split,found_split,gmm_bic,score_diff,true_score_diff,num_iter,initial_split,per_intv,parent_intv,cc_ari,gmm_ari,gmm_ari_res,kmeans_ari,kmeans_ari_res,spectral_ari,spectral_ari_res,cc_nmi,gmm_nmi,gmm_nmi_res,kmeans_nmi,kmeans_nmi_res,spectral_nmi,spectral_nmi_res,cc_fmi,gmm_fmi,gmm_fmi_res,kmeans_fmi,kmeans_fmi_res,spectral_fmi,spectral_fmi_res\n")

        #headers = [i for i in range(0, self.V)]                                !! NO IDEA !!
        # Initialize the logger
        # logger = Logger(self.log_path, log_to_disk=self.log_flag, verbose=self.verbose)
        # logger.Begin()
        # logger.WriteLog("BEGIN LOGGING FOR FILE: " + self.filename)

        dims = self.gt.shape[1]
        g = Graph(dims)

        #if not needed_nodes:
        # Create Edges based on ground truth adjacency matrix
        self.Edges = [[None for _ in range(self.V)] for _ in range(self.V)]

        # Construct the Edges from the ground truth (gt_network)
        for i in range(dims):
            for j in range(dims):
                if self.gt[i, j] == 1:
                    g.addEdge(i, j)
                    self.Edges[i][j] = Edge(i, j, [], 0)
        self.ordering = g.nonRecursiveTopologicalSort()
        #else:
        #self.ordering = needed_nodes
        print('ORDERING ' + str(self.ordering))
        print(f"Interventions {self.intv}")
        # Create Node objects from normalized variables
        for node in range(0, self.V):
            self.Nodes.append(Node(self.vars[:, node].reshape(self.vars.shape[0], -1), self))

        ari_scores = []
        nmi_scores = []
        fmi_scores = []

        for i,variable_index in enumerate(self.ordering):
            initial_split = 0
            if needed_nodes and variable_index not in needed_nodes: continue
            is_intv = 1 if variable_index in self.intv else 0
            is_intv_found = 0
            parent_intv = 0
            # Get parents of the node
            pa_i = np.where(self.gt[:, variable_index] == 1)[0]
            for parent in pa_i:
                if parent in self.foundIntv:
                    parent_intv = 1

            print('ITER ' + str(i))
            # If node has no parents print and skip it
            if len(pa_i) == 0:
                print(f"Node {variable_index} has no parents.")
                continue
            print(f"NODE {variable_index} has parents {pa_i}")

            X = self.vars[:, pa_i]
            y = self.vars[:, variable_index]
            data_xy = np.hstack((X, y.reshape(-1, 1)))  # Combine X and y for gmm, kmeans and spectral clustering

            # Fit MARS on the entire dataset
            sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X, y)
            y_pred = rf.predict_mars(X, rearth)
            residuals = y - y_pred

            # Score for ONE MODEL
            cost_all = self.ComputeScore(hinge_count, interactions, sse, score, len(y), self.Nodes[i].min_diff,
                                         np.array([len(pa_i)]), show_graph=False)
            self.score_all[variable_index] = cost_all
            print(f"Cost of ONE MODEL : {cost_all}")
            final_cost = cost_all
            min_cost = math.inf
            best_k = 1
            final_labels = np.zeros(self.vars.shape[0])
            true_k = int(self.attributes[4]) + 1
            labels_true = np.array([0] * int(self.attributes[2]) +
                                   [i for i in range(1, true_k) for _ in range(int(self.attributes[3]))])
            #print(labels_true)
            # Calculating True cost gain if there were interventions
            true_cost_gain = 0
            if is_intv:
                scores = []
                hinge_counts = []
                interactions = []
                sse_values = []
                row_counts = []

                for cluster in range(true_k):
                    X_group = X[labels_true == cluster, :]
                    y_group = y[labels_true == cluster]

                    # Fit spline model for each group
                    sse, score, coeff, hinge_count, interaction, _ = self.slope_.FitSpline(X_group, y_group)

                    scores.append(score)
                    hinge_counts.append(hinge_count)
                    interactions.append(interaction)
                    sse_values.append(sse)
                    row_counts.append(len(X_group))

                # Compute split cost
                true_cost_split = self.ComputeScoreSplit(
                    hinge_counts, interactions, sse_values, scores, row_counts,
                    self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False
                )
                print(f"True Cost Split: {true_cost_split}")
                true_cost_gain = cost_all - true_cost_split

            min_cc_ari = 0
            min_cc_nmi = 0
            min_cc_fmi = 0
            for clusters in range(2,k+1):
                per_intv = 0
                gmm_ari = 0
                kmeans_ari = 0
                spectral_ari = 0
                gmm_ari_res = 0
                kmeans_ari_res = 0
                spectral_ari_res = 0
                gmm_nmi = 0
                kmeans_nmi = 0
                spectral_nmi = 0
                gmm_nmi_res = 0
                kmeans_nmi_res = 0
                spectral_nmi_res = 0
                gmm_fmi = 0
                kmeans_fmi = 0
                spectral_fmi = 0
                gmm_fmi_res = 0
                kmeans_fmi_res = 0
                spectral_fmi_res = 0
                is_intv_found_k = 0
                is_intv_k = 1 if ((variable_index in self.intv) and (clusters == true_k)) else 0
                cc_ari = 0
                cc_nmi = 0
                cc_fmi = 0

                # Initialize predicted labels with empty lists
                gmm_labels = []
                kmeans_labels = []
                spectral_labels = []
                gmm_res_labels = []
                kmeans_res_labels = []
                spectral_res_labels = []

                # Get the clustering results ( accuracy and f1_score) of GMM , Kmeans, Spectral on Xy-data and on residuals
                if is_intv:
                    tc = TraditionalClustering()
                    clustering_results, predicted_labels = tc.getTraditionalClustering(data_xy, residuals, labels_true, clusters)
                    (
                        gmm_ari, kmeans_ari, spectral_ari,
                        gmm_ari_res, kmeans_ari_res, spectral_ari_res,
                        gmm_nmi, kmeans_nmi, spectral_nmi,
                        gmm_nmi_res, kmeans_nmi_res, spectral_nmi_res,
                        gmm_fmi, kmeans_fmi, spectral_fmi,
                        gmm_fmi_res, kmeans_fmi_res, spectral_fmi_res
                    ) = (
                        clustering_results.get(key, 0) for key in [
                        "gmm_xy_ARI", "kmeans_xy_ARI", "spectral_xy_ARI",
                        "gmm_res_ARI", "kmeans_res_ARI", "spectral_res_ARI",
                        "gmm_xy_NMI", "kmeans_xy_NMI", "spectral_xy_NMI",
                        "gmm_res_NMI", "kmeans_res_NMI", "spectral_res_NMI",
                        "gmm_xy_FMI", "kmeans_xy_FMI", "spectral_xy_FMI",
                        "gmm_res_FMI", "kmeans_res_FMI", "spectral_res_FMI"
                    ]
                    )
                    # Unpack predicted labels
                    (
                        gmm_labels, kmeans_labels, spectral_labels,
                        gmm_res_labels, kmeans_res_labels, spectral_res_labels
                    ) = (
                        predicted_labels.get(key, []) for key in [
                        "gmm_xy", "kmeans_xy", "spectral_xy",
                        "gmm_res", "kmeans_res", "spectral_res"
                    ]
                    )

                # Get the cost for splitting with k = clusters (WHEN POSSIBLE)
                cost_split, labels_split, k_split_possible, num_iter, gmm_bic, initial_split = self.my_function(variable_index, pa_i, residuals, clusters, random, mdl_th, needed_nodes)

                if k_split_possible:
                    # compair scores and all that and save node row to the file.
                    print(f'COST for splitting model with {clusters} is {str(cost_split)}')
                    eps = 0  # 3100  # THREASHOLD FOR MDL DECISION (BITS)

                    if cost_split < cost_all:
                        print(f"Splitting model with {clusters} is better with score  {cost_all - cost_split}")
                        #final_cost = cost_split
                        #self.foundIntv.append(variable_index)
                        is_intv_found_k = 1
                        is_intv_found = 1
                        # Compute CC method clustering metrics
                        cc_ari = adjusted_rand_score(labels_true, labels_split)
                        cc_nmi = normalized_mutual_info_score(labels_true, labels_split)
                        cc_fmi = fowlkes_mallows_score(labels_true, labels_split)
                        #print("CC ARI:", cc_ari)
                        #print("CC NMI:", cc_nmi)
                        #print("CC FMI:", cc_fmi)
                    else:
                        print(
                            f"Original model is better than {clusters} split, with cost difference {cost_split - cost_all}")
                    if cost_split < min_cost:
                        min_cost = cost_split
                        best_k = clusters
                        final_labels = labels_split
                        min_cc_ari = cc_ari
                        min_cc_nmi = cc_nmi
                        min_cc_fmi = cc_fmi
                    cluster_sizes = np.zeros(clusters)
                    for j in labels_split:
                        cluster_sizes[j] += 1
                    per_intv = ((self.vars.shape[0] - max(cluster_sizes)) / self.vars.shape[0]) * 100
                else: print(f"Original model is better with cost {cost_all}  SPLITTING with {clusters} clusters NOT POSSIBLE")

                if needed_nodes:
                    # PLOTING THE FINAL RESULT IF ONLY 1 PARENT
                    if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
                        #self.plot.plot_2d_results(only_one, pa_i, variable_index, final_labels, y_pred, y_pred1,
                         #                         y_pred2, labels_true)
                        self.plot.plot_2d_other(pa_i, variable_index, final_labels, 'CC method')
                        self.plot.plot_2d_other(pa_i, variable_index, kmeans_labels, "KMeans")
                        self.plot.plot_2d_other(pa_i, variable_index, kmeans_res_labels, "KMeans on residuals")
                        self.plot.plot_2d_other(pa_i, variable_index, gmm_labels, "GMM")
                        self.plot.plot_2d_other(pa_i, variable_index, gmm_res_labels, "GMM on residuals")
                        self.plot.plot_2d_other(pa_i, variable_index, spectral_labels, "Spectral")
                        self.plot.plot_2d_other(pa_i, variable_index, spectral_res_labels, "Spectral on residuals")

                    # PLOT THE FINAL RESULT IF 2 PARENTS
                    if len(pa_i) == 2:  # Only plot when there are two parents (3D case)
                        self.plot.plot_3d_other(pa_i, variable_index, final_labels, 'CC method')
                        self.plot.plot_3d_other(pa_i, variable_index, kmeans_labels, "KMeans")
                        self.plot.plot_3d_other(pa_i, variable_index, kmeans_res_labels, "KMeans on residuals")
                        self.plot.plot_3d_other(pa_i, variable_index, gmm_labels, "GMM")
                        self.plot.plot_3d_other(pa_i, variable_index, gmm_res_labels, "GMM on residuals")
                        self.plot.plot_3d_other(pa_i, variable_index, spectral_labels, "Spectral")
                        self.plot.plot_3d_other(pa_i, variable_index, spectral_res_labels, "Spectral on residuals")
                print(f"{cost_split}")
                if cost_split is math.inf:
                    score_diff = 0
                else:
                    score_diff = cost_all - cost_split
                if not needed_nodes:
                    # id, num_parents, true_split, found_split, score_diff
                    with open(nodestats_file, "a") as stats:
                        stats.write(
                            f"{variable_index},{len(pa_i)},{clusters},{is_intv_k},{is_intv_found_k},{gmm_bic},{int(score_diff)},{int(true_cost_gain)},{num_iter},{initial_split},{per_intv},{parent_intv},{cc_ari},{gmm_ari},{gmm_ari_res},{kmeans_ari},{kmeans_ari_res},{spectral_ari},{spectral_ari_res},{cc_nmi},{gmm_nmi},{gmm_nmi_res},{kmeans_nmi},{kmeans_nmi_res},{spectral_nmi},{spectral_nmi_res},{cc_fmi},{gmm_fmi},{gmm_fmi_res},{kmeans_fmi},{kmeans_fmi_res},{spectral_fmi},{spectral_fmi_res}\n")

                else: print(f"INITIAL SPLIT :  {initial_split}")
            if min_cost is math.inf:
                self.score_all[variable_index] = cost_all
                min_cost = cost_all
            else:
                self.score_split[variable_index] = min_cost
            #print(f"Cost of one model is {cost_all} and Min cost is {min_cost}")
            print(f"Best k is : {best_k}")
            self.node_labels[variable_index] = final_labels
            self.k[variable_index] = best_k
            nmi_scores.append(min_cc_nmi)
            fmi_scores.append(min_cc_fmi)
            if is_intv_found:
                self.foundIntv.append(variable_index)
                self.split[variable_index] = 1
                ari_scores.append(min_cc_ari)
        labels_file = f"{self.filename}/node_labels.txt"
        np.savetxt(labels_file, self.node_labels, fmt="%d")
        return self.foundIntv, ari_scores  # ari only for nodes where foundIntv is true

    def my_function(self, i, pa_i, residuals, k, random, mdl_th, needed_nodes):
        X = self.vars[:, pa_i]
        y = self.vars[:, i]
        best_cost = math.inf
        best_result = None
        # Fit GMM with 1 component
        gmm1 = GaussianMixture(n_components=1, random_state=19)
        gmm1.fit(residuals.reshape(-1, 1))

        # Fit GMM with k components
        gmm_k = GaussianMixture(n_components=k, random_state=19)
        gmm_k.fit(residuals.reshape(-1, 1))
        initial_labels = gmm_k.predict(residuals.reshape(-1, 1))
        # spec_k = SpectralClustering(n_clusters=k, random_state=19, affinity='nearest_neighbors')
        # initial_labels = spec_k.fit_predict(residuals.reshape(-1, 1))
        # Get BIC scores
        bic1 = gmm1.bic(residuals.reshape(-1, 1))
        bic2 = gmm_k.bic(residuals.reshape(-1, 1))

        # Compare BIC scores
        gmm_bic = 0 if bic1 <= bic2 else 1
        def quantile_split(residuals, k):
            quantiles = np.percentile(residuals, np.linspace(0, 100, k + 1))
            return np.digitize(residuals, quantiles[1:-1])

        def random_split(residuals, k):
            return np.random.choice(k, size=residuals.shape[0])

        def gmm_split(residuals, k):
            gmm = GaussianMixture(n_components=k, random_state=19)
            gmm.fit(residuals.reshape(-1, 1))
            return gmm.predict(residuals.reshape(-1, 1))
        initial_split = 0
        for init_type in ['gmm']: #, 'random', 'quantile'
            if init_type == 'gmm':
                initial_labels = gmm_split(residuals, k)
                initial_split = 0
            '''elif init_type == 'random':
                initial_labels = random_split(residuals, k)
                initial_split = 1'''
            '''elif init_type == 'quantile':   # Spectral on residuals .
                initial_labels = quantile_split(residuals, k)
                initial_split = 2'''

            first_iter = True
            num_iter = 0
            last_groups = initial_labels.copy()
            prev_cost_split = math.inf
            labels = initial_labels.copy()

            while True:
                num_iter += 1
                mars_models = {}
                group_residuals = {}
                hinge_counts = [] #np.zeros(k)
                interactions = []
                sse_values = []
                scores = []
                group_sizes = [] #np.zeros(k)
                k_split_possible = True

                for cluster in range(k):
                    X_group = X[labels == cluster, :]
                    y_group = y[labels == cluster]

                    if X_group.shape[0] <= 1:
                        k_split_possible = False
                        break

                    sse, score, coeff, hinge_count, interaction, rearth = self.slope_.FitSpline(X_group, y_group)
                    mars_models[cluster] = rearth
                    hinge_counts.append(hinge_count)
                    interactions.append(interaction)
                    sse_values.append(sse)
                    scores.append(score)
                    group_sizes.append(len(y_group))

                #if not k_split_possible:
                #    continue

                for j in range(X.shape[0]):
                    x = X[j, :]
                    y_actual = y[j]
                    residuals_c = {
                        c: abs(y_actual - rf.predict_mars(x, mars_models[c]))
                        if c in mars_models else float('inf')
                        for c in range(k)
                    }
                    labels[j] = min(residuals_c, key=residuals_c.get)

                changes = np.sum(labels != last_groups)
                if (not first_iter and (changes / len(labels) < 0.005)) or num_iter > 100:
                    final_labels = labels.copy()
                    break

                if mdl_th:
                    cur_cost_split = self.ComputeScoreSplit(
                        hinge_counts, interactions, sse_values, scores,
                        group_sizes, self.Nodes[i].min_diff, np.array([len(pa_i)]),
                        show_graph=False
                    )
                    if cur_cost_split > prev_cost_split:
                        final_labels = last_groups.copy()
                        break
                    prev_cost_split = cur_cost_split

                first_iter = False
                last_groups = labels.copy()

            # Final model fit for scoring
            sse_list, score_list, hinge_counts_list, interactions_list, final_group_sizes = [],[],[], [], []
            mars_models = {}

            for cluster in range(k):
                X_group = X[final_labels == cluster, :]
                y_group = y[final_labels == cluster]

                if X_group.shape[0] > 1:
                    sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X_group, y_group)
                    sse_list.append(sse)
                    score_list.append(score)
                    hinge_counts_list.append(hinge_count)
                    interactions_list.append(interactions)
                    final_group_sizes.append(len(y_group))
                    final_group_sizes.append(len(y_group))
                    mars_models[cluster] = rearth

            cost_split = self.ComputeScoreSplit(
                hinge_counts_list, interactions_list, sse_list, score_list,
                final_group_sizes, self.Nodes[i].min_diff, np.array([len(pa_i)]),
                show_graph=False
            )
            print(f"COST SPLIT {cost_split}")
            if cost_split < best_cost:
                best_cost = cost_split
                best_result = (cost_split, final_labels, k_split_possible, num_iter, gmm_bic, initial_split)

        return best_result

    # SCORE COMPUTATION PART
    def ComputeScore(self, hinges, interactions, sse, model, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        model_cost = self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        residuals_cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff)
        models = model + model_cost
        cost = (residuals_cost + base_cost + models)
        return cost;

    def ComputeScoreSplit(self, hinges, interactions, sse, models, rows, mindiff, m, show_graph=False):
        base_cost = self.slope_.model_score(m) + m * np.log2(self.V)  # m is the number of parent variables
        # Initialize total model cost and residuals cost
        total_model_cost = 0
        total_residuals_cost = 0
        total_rows = 0
        # Iterate over each cluster
        for i in range(len(hinges)):
            # Calculate model cost for the current cluster
            model_cost = self.slope_.model_score(hinges[i]) + self.AggregateHinges(interactions[i], m)
            #print(f'MODEL COST: {model_cost}')
            total_model_cost += model_cost
            # Calculate residuals cost for the current cluster
            residuals_cost = self.slope_.gaussian_score_emp_sse(sse[i], rows[i], mindiff)
            #print(f'RESIDUAL COST: {residuals_cost}')
            total_residuals_cost += residuals_cost
            total_rows += rows[i]
        #print('Total ROWS: ' + str(total_rows)) # Print total rows for debugging
        labels_cost = total_rows * math.log2(len(hinges))
        total_cost = base_cost + total_residuals_cost + total_model_cost + labels_cost
        return total_cost

    def OldComputeScore(self, source, target, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        sse, model, coeffs, hinges, interactions, rearth= self.slope_.FitSpline(source, target, self.M, show_graph);
        base_cost = base_cost + self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff) + model + base_cost;
        return cost, coeffs;

    def AggregateHinges(self, hinges, k):
        cost = 0;
        flag = 1;

        for M in hinges:
            cost += self.slope_.logN(M) + Combinator(M, k) + M * np.log2(self.F);
        return cost;





