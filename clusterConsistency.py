from node import Node;
from edge import Edge;
from slope import Slope;
from utils import *
from logger import Logger
import numpy as np;
import time
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
import random
class ClusterConsistency:
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
        self.all_labels = []
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
        self.found_intvs = []
        self.true_intvs = []
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
            labels_file = f"{self.filename}/node_labels.txt"
            stats_file = f"{self.filename}/node_STATS.txt"
            self.gt = np.loadtxt(gt_file, delimiter=',')
            data = np.loadtxt(data_file, delimiter=',')
            #self.all_labels = np.loadtxt(labels_file, delimiter=',')
            node_stats = np.genfromtxt(stats_file, delimiter=',', skip_header=1)
            # Column indices based on your header description
            ID_COL = 0
            TRUE_SPLIT_COL = 3
            FOUND_SPLIT_COL = 4
            ARI_COL = 9

            # Extract sets of node ids
            self.found_intvs = set(int(row[ID_COL]) for row in node_stats if row[FOUND_SPLIT_COL] == 1)
            self.true_intvs = set(int(row[ID_COL]) for row in node_stats if row[TRUE_SPLIT_COL] == 1)
            self.best_intv = set(int(row[ID_COL]) for row in node_stats if row[ARI_COL] >= 0.9)
            print(f"GOOD: {self.best_intv}")
            self.bad_intv = set(int(row[ID_COL]) for row in node_stats if ((row[ARI_COL] < 0.8) & (row[ARI_COL] > 0.2)))
            print(f"BAD: {self.bad_intv}")
            if not self.best_intv:
                print("No high-ARI nodes found.")
                return
            if not self.bad_intv:
                print("No low-ARI nodes found.")
                return
            try:
                intvs = np.loadtxt(intv_file, delimiter=',', dtype=int)
            except ValueError as e:
                intvs = []  # Or handle as needed
            self.intv = intvs
            self.vars = data
            self.plot = Plotting(self.vars)
            print(f"Data shape: {data.shape}")
            self.V = self.vars.shape[1]
            print(f"self.V: {self.V}")
            self.result = np.zeros((self.V, self.vars.shape[0]))
            self.node_labels = np.zeros((self.V, self.vars.shape[0]))
            self.split = np.zeros(self.V)
            self.k = np.zeros(self.V)
            self.score_all = np.zeros(self.V)
            self.score_split = np.zeros(self.V);
            # Load Attributes
            attributes_file = f"{self.filename}/attributes1.txt"
            with open(attributes_file, "r") as atts:
                lines = atts.readlines()
                values = lines[1].strip()  # Second line contains the values
                # Convert the values to a list
                self.attributes = values.split(", ")
        except Exception as e:
            print(f"An error occurred: {e}")

    def run(self):

        dims = self.gt.shape[1]
        g = Graph(dims)

        # Create Edges based on ground truth adjacency matrix
        self.Edges = [[None for _ in range(self.V)] for _ in range(self.V)]
        if self.V != dims:
            raise ValueError(f"Mismatch: data has {self.V} variables, but ground truth expects {dims} nodes.")
        # Construct the Edges from the ground truth (gt_network)
        print(f"DIMS : {dims}")
        for i in range(dims):
            for j in range(dims):
                if self.gt[i, j] == 1:
                    g.addEdge(i, j)
                    print(f"Edge {i}:{j}")
                    self.Edges[i][j] = Edge(i, j, [], 0)
        self.ordering = g.nonRecursiveTopologicalSort()

        print('ORDERING ' + str(self.ordering))
        print(f"Interventions {self.intv}")
        # Create Node objects from normalized variables
        for node in range(0, self.V):
            self.Nodes.append(Node(self.vars[:, node].reshape(self.vars.shape[0], -1), self))

        self.correct_intvs = self.found_intvs & self.true_intvs
        self.false_intvs = self.true_intvs - self.found_intvs
        good_intv = random.choice(list(self.best_intv))

        pa_g = np.where(self.gt[:, good_intv] == 1)[0]
        X_g = self.vars[:, pa_g]
        y_g = self.vars[:, good_intv]

        # Fit MARS on the entire dataset
        sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X_g, y_g)
        y_pred_g = rf.predict_mars(X_g, rearth)
        residuals_g = y_g - y_pred_g
        #labels_good = self.all_labels[good_intv]
        min_dif_g = self.Nodes[good_intv].min_diff
        cost_split, labels_good, k_split_possible, num_iter, gmm_bic, initial_split = self.my_function(good_intv,
                                                                                                        pa_g, residuals_g,
                                                                                                        2,
                                                                                                        False, False,
                                                                                                        [],
                                                                                                        min_dif_g)
        self.plot.plot_3d_other(pa_g, good_intv, labels_good, 'Found Split')

        labels_true = np.array([0] * int(self.attributes[2]) +
                               [i for i in range(1, 2) for _ in range(int(self.attributes[3]))])

        for i,variable_index in enumerate(self.bad_intv):
            if variable_index == good_intv: continue
            # Get parents of the node
            pa_i = np.where(self.gt[:, variable_index] == 1)[0]

            X = self.vars[:, pa_i]
            y = self.vars[:, variable_index]
            #labels_g = self.all_labels[good_intv]
            #labels_i = self.all_labels[variable_index]
            labels_g = labels_good

            # Fit MARS on the entire dataset
            sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X, y)
            y_pred = rf.predict_mars(X, rearth)
            residuals = y - y_pred

            # Score for MODEL
            cost_all = self.ComputeScore(hinge_count, interactions, sse, score, len(y), self.Nodes[i].min_diff,
                                         np.array([len(pa_i)]), show_graph=False)
            self.score_all[variable_index] = cost_all
            print(f"Cost of ONE MODEL : {cost_all}")
            min_dif = self.Nodes[variable_index].min_diff
            cost_split, labels_i, k_split_possible, num_iter, gmm_bic, initial_split = self.my_function(variable_index,
                                                                                                           pa_i,
                                                                                                           residuals,
                                                                                                           2,
                                                                                                           False, False,
                                                                                                           [],
                                                                                                           min_dif)
            # Calculating True cost gain if there were interventions
            my_split_data = {}

            for cluster_label in np.unique(labels_g):
                indices = np.where(labels_g == cluster_label)[0]
                X_group = X[indices, :]
                y_group = y[indices]
                sse, score, coeff, hinge_count, interaction, _ = self.slope_.FitSpline(X_group, y_group)
                my_split_data[cluster_label] = {'sse': sse, 'hinge_count': hinge_count,
                                                'interaction': interaction, 'score': score,
                                                'row_count': len(X_group)}

            # Ensure consistent ordering of keys when passing to ComputeScoreSplit
            sorted_labels = sorted(my_split_data.keys())
            new_cost_split = self.ComputeScoreSplit(
                [my_split_data[label]['hinge_count'] for label in sorted_labels],
                [my_split_data[label]['interaction'] for label in sorted_labels],
                [my_split_data[label]['sse'] for label in sorted_labels],
                [my_split_data[label]['score'] for label in sorted_labels],
                [my_split_data[label]['row_count'] for label in sorted_labels],
                min_dif, np.array([len(pa_i)]), show_graph=False
            )
            #cost_gain = cost_all - cost_split
            print(f"COST OLD : {cost_split}")
            print(f"COST NEW : {new_cost_split}")
            ari_old = adjusted_rand_score(labels_true, labels_i)
            ari_new = adjusted_rand_score(labels_true, labels_g)
            print(f"ARI OLD : {ari_old}")
            print(f"ARI NEW : {ari_new}")
            self.plot.plot_3d_other(pa_i, variable_index, labels_i, 'Old Split')
            self.plot.plot_3d_other(pa_i, variable_index, labels_g, 'New Split')

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
            model_cost = self.slope_.model_score(hinges[i]) + self.AggregateHinges(interactions[i], m) + models[i]
            total_model_cost += model_cost
            # Calculate residuals cost for the current cluster
            residuals_cost = self.slope_.gaussian_score_emp_sse(sse[i], rows[i], mindiff)
            total_residuals_cost += residuals_cost
            total_rows += rows[i]

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


    def my_function(self, i, pa_i, residuals, k, random, mdl_th, needed_nodes, min_dif):
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
            if random:
                initial_labels = random_split(residuals, k)
                initial_split = 1

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
                        group_sizes, min_dif, np.array([len(pa_i)]),
                        show_graph=False
                    )
                    if cur_cost_split > prev_cost_split:
                        final_labels = last_groups.copy()
                        break
                    prev_cost_split = cur_cost_split

                first_iter = False
                last_groups = labels.copy()

            # Final model fit for scoring
            '''sse_list, score_list, hinge_counts_list, interactions_list, final_group_sizes = [],[],[], [], []
            mars_models = {}
            print(f"Calculated K is {k}")
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
                    mars_models[cluster] = rearth

            cost_split = self.ComputeScoreSplit(
                hinge_counts_list, interactions_list, sse_list, score_list,
                final_group_sizes, self.Nodes[i].min_diff, np.array([len(pa_i)]),
                show_graph=False
            )'''
            calculated_split_data = {}
            for cluster_label in np.unique(final_labels):
                indices = np.where(final_labels == cluster_label)[0]
                X_group = X[indices, :]
                y_group = y[indices]
                if X_group.shape[0] > 1:
                    sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X_group, y_group)
                    calculated_split_data[cluster_label] = {'sse': sse, 'hinge_count': hinge_count,
                                                            'interaction': interactions, 'score': score,
                                                            'row_count': len(y_group)}

            # Ensure consistent ordering of keys when passing to ComputeScoreSplit
            sorted_calculated_labels = sorted(calculated_split_data.keys())
            cost_split = self.ComputeScoreSplit(
                [calculated_split_data[label]['hinge_count'] for label in sorted_calculated_labels],
                [calculated_split_data[label]['interaction'] for label in sorted_calculated_labels],
                [calculated_split_data[label]['sse'] for label in sorted_calculated_labels],
                [calculated_split_data[label]['score'] for label in sorted_calculated_labels],
                [calculated_split_data[label]['row_count'] for label in sorted_calculated_labels],
                min_dif, np.array([len(pa_i)]), show_graph=False
            )
            #print(f"COST SPLIT (Modified): {cost_split}")
            print(f"COST SPLIT {cost_split}")
            if cost_split < best_cost:
                best_cost = cost_split
                best_result = (cost_split, final_labels, k_split_possible, num_iter, gmm_bic, initial_split)

        return best_result


