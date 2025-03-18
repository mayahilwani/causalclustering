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
import seaborn as sns
from combinator import Combinator
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score
import math
from plotting import Plotting
from traditionalClustering import TraditionalClustering as tc

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
            try:
                intvs = np.loadtxt(intv_file, delimiter=',', dtype=int)
            except ValueError as e:
                intvs = []  # Or handle as needed
            self.vars = data
            self.plot = Plotting(self.vars)
            # Standardize the loaded data in self.vars
            #normalized_vars = self.vars  # Standardize(self.vars)
            self.V = self.vars.shape[1]
            self.result = np.zeros((self.V, self.vars.shape[0]))
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
                stats.write("id, num_parents, k, true_split, found_split, gmm_bic, score_diff, true_score_diff, num_iter, method_acc, gmm_acc, gmm_acc_res, kmeans_acc, kmeans_acc_res, spectral_acc, spectral_acc_res, f1, gmm_f1, gmm_f1_res, kmeans_f1, kmeans_f1_res, spectral_f1, spectral_f1_res\n")

        #headers = [i for i in range(0, self.V)]                                !! NO IDEA !!
        # Initialize the logger
        # logger = Logger(self.log_path, log_to_disk=self.log_flag, verbose=self.verbose)
        # logger.Begin()
        # logger.WriteLog("BEGIN LOGGING FOR FILE: " + self.filename)

        dims = self.gt.shape[1]
        g = Graph(dims)

        if not needed_nodes:
            # Create Edges based on ground truth adjacency matrix
            self.Edges = [[None for _ in range(self.V)] for _ in range(self.V)]

            # Construct the Edges from the ground truth (gt_network)
            for i in range(dims):
                for j in range(dims):
                    if self.gt[i, j] == 1:
                        g.addEdge(i, j)
                        self.Edges[i][j] = Edge(i, j, [], 0)
            self.ordering = g.nonRecursiveTopologicalSort()
        else:
            self.ordering = needed_nodes
        print('ORDERING ' + str(self.ordering))
        print(f"Interventions {self.intv}")
        # Create Node objects from normalized variables
        for node in range(0, self.V):
            self.Nodes.append(Node(self.vars[:, node].reshape(self.vars.shape[0], -1), self))

        accuracies = []

        for i,variable_index in enumerate(self.ordering):
            is_intv = 1 if variable_index in self.intv else 0
            # Get parents of the node
            pa_i = np.where(self.gt[:, variable_index] == 1)[0]
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
            self.scoref[variable_index] = cost_all
            final_cost = cost_all
            min_cost = math.inf
            best_k = 1
            final_labels = np.zeros(self.vars.shape[0])
            true_k = self.attributes[4] + 1
            labels_true = np.array([0] * int(self.attributes[2]) +
                                   [i for i in range(1, true_k) for _ in range(int(self.attributes[3]))])

            # Calculating True cost gain if there were interventions
            true_cost_gain = 0
            if is_intv:
                scores = []
                hinge_counts = []
                interactions = []
                sse_values = []
                row_counts = []

                for cluster in true_k:
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

                true_cost_gain = cost_all - true_cost_split

            for clusters in range(2,k+1):
                gmm_acc = 0
                kmeans_acc = 0
                gmm_acc_res = 0
                kmeans_acc_res = 0
                spectral_acc = 0
                spectral_acc_res = 0
                cc_accuracy = 0
                cc_f1 = 0
                gmm_f1 = 0
                gmm_f1_res = 0
                kmeans_f1 = 0
                kmeans_f1_res = 0
                spectral_f1 = 0
                spectral_f1_res = 0
                is_intv_found_k = 0
                is_intv_k = 1 if ((variable_index in self.intv) and (clusters == true_k)) else 0

                # Get the clustering results ( accuracy and f1_score) of GMM , Kmeans, Spectral on Xy-data and on residuals
                if is_intv:
                    clustering_results = tc.getTraditionalClustering(data_xy, residuals, labels_true, clusters)
                    (
                        gmm_acc, kmeans_acc, spectral_acc,
                        gmm_acc_res, kmeans_acc_res, spectral_acc_res,
                        gmm_f1, kmeans_f1, spectral_f1,
                        gmm_f1_res, kmeans_f1_res, spectral_f1_res
                    ) = (
                        clustering_results.get(key, 0) for key in [
                        "gmm_xy_accuracy", "kmeans_xy_accuracy", "spectral_xy_accuracy",
                        "gmm_res_accuracy", "kmeans_res_accuracy", "spectral_res_accuracy",
                        "gmm_xy_f1", "kmeans_xy_f1", "spectral_xy_f1",
                        "gmm_res_f1", "kmeans_res_f1", "spectral_res_f1"
                    ]
                    )

                # Get the cost for splitting with k = clusters (WHEN POSSIBLE)
                cost_split, labels_split, k_split_possible, num_iter, gmm_bic = self.my_function(variable_index, pa_i, residuals, clusters, random, mdl_th, needed_nodes)

                if k_split_possible:
                    # compair scores and all that and save node row to the file.
                    print('COST for splitting model is ' + str(cost_split))
                    eps = 0  # 3100  # THREASHOLD FOR MDL DECISION (BITS)

                    if cost_split < min_cost:
                        min_cost = cost_split
                        best_k = clusters
                        final_labels = labels_split

                    if cost_split < cost_all:
                        print(f"Splitting model with {clusters} is better with score  {cost_all - cost_split}")
                        #final_cost = cost_split
                        #self.foundIntv.append(variable_index)
                        is_intv_found_k = 1

                        re_labels = tc.match_labels_sklearn(labels_split, labels_true)
                        ami_score = adjusted_mutual_info_score(labels_true, re_labels)
                        print("Adjusted Mutual Information Score:", ami_score)

                        cc_f1 = tc.compute_f1(re_labels, labels_true)      # ALSO HAVE F1_SCORES ARRAY TO RETURN??
                        cc_accuracy = accuracy_score(labels_true, re_labels)
                        print(f"Accuracy: {cc_accuracy}")
                        accuracies.append(cc_accuracy)
                    else: print(f"Original model is better than {clusters} split, with cost difference {cost_split - cost_all}")
                else: print(f"Original model is better with cost {cost_all}  SPLITTING with {clusters} clusters NOT POSSIBLE")

                score_diff = cost_all - cost_split
                if not needed_nodes:
                    # id, num_parents, true_split, found_split, score_diff
                    with open(nodestats_file, "a") as stats:
                        stats.write(
                            f"{variable_index},{len(pa_i)}, {clusters}, {is_intv_k},{is_intv_found_k},{gmm_bic},{score_diff},{true_cost_gain},{num_iter},{cc_accuracy},{gmm_acc},{gmm_acc_res},{kmeans_acc},{kmeans_acc_res},{spectral_acc},{spectral_acc_res},{cc_f1},{gmm_f1},{gmm_f1_res},{kmeans_f1},{kmeans_f1_res},{spectral_f1},{spectral_f1_res} \n")

                # "id, num_parents, k, true_split, found_split, gmm_bic, score_diff, true_score_diff, num_iter, method_acc, gmm_acc, gmm_acc_res, kmeans_acc, kmeans_acc_res, spectral_acc, spectral_acc_res, f1, gmm_f1, gmm_f1_res, kmeans_f1, kmeans_f1_res, spectral_f1, spectral_f1_res
                #######################################################################################
                # need to write values to file.
                self.score_all[variable_index] = cost_all
                self.score_split[variable_index] = min_cost
                self.node_labels[variable_index] = final_labels
                self.k[variable_index] = best_k


        # Plot when needed nodes only for k that had best score. ( OR FOR THE TRUE k ??? )
        '''            if needed_nodes:
                # PLOTING THE FINAL RESULT IF ONLY 1 PARENT
                if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
                    self.plot.plot_2d_results(only_one, pa_i, variable_index, final_labels, y_pred, y_pred1, y_pred2, labels_true)
                    self.plot.plot_2d_other(pa_i, variable_index, kmeans_labels_xy, "KMeans")
                    self.plot.plot_2d_other(pa_i, variable_index, labels_xy, "GMM")
                    self.plot.plot_2d_other(pa_i, variable_index, spectral_labels_xy, "Spectral")

                # PLOT THE FINAL RESULT IF 2 PARENTS
                if len(pa_i) == 2:  # Only plot when there are two parents (3D case)
                    self.plot.plot_3d_results(only_one, pa_i, variable_index, final_labels, rearth, y_pred1, y_pred2, labels_true)
                    self.plot.plot_3d_other(pa_i, variable_index, kmeans_labels_xy, "KMeans")
                    self.plot.plot_3d_other(pa_i, variable_index, labels_xy, "GMM")
                    self.plot.plot_3d_other(pa_i, variable_index, spectral_labels_xy, "Spectral")'''






    def my_function(self, i, pa_i, residuals, k, random, mdl_th, needed_nodes):
        X = self.vars[:, pa_i]
        y = self.vars[:, i]

        # Fit GMM with 1 component
        gmm1 = GaussianMixture(n_components=1, random_state=19)
        gmm1.fit(residuals.reshape(-1, 1))

        # Fit GMM with k components
        gmm_k = GaussianMixture(n_components=k, random_state=19)
        gmm_k.fit(residuals.reshape(-1, 1))
        initial_labels = gmm_k.predict(residuals.reshape(-1, 1))

        # Get BIC scores
        bic1 = gmm1.bic(residuals.reshape(-1, 1))
        bic2 = gmm_k.bic(residuals.reshape(-1, 1))

        # Compare BIC scores
        gmm_bic = 0 if bic1 <= bic2 else 1                  # NEEDED TO RETURN (save to file)

        if random:
            initial_labels = np.random.choice(np.arrange(k), size=residuals.shape[0])

        first_iter = True
        num_iter = 0
        last_groups = initial_labels.copy()
        prev_cost_split = math.inf
        labels = initial_labels.copy()

        while True:
            num_iter += 1

            # Fit different MARS lines for each of the k groups
            mars_models = {}  # Store MARS models
            group_residuals = {}  # Store residuals
            hinge_counts = []
            interactions = []
            sse_values = []
            scores = []
            group_sizes = []
            k_split_possible = True

            for cluster in range(k):
                empty_group = False
                X_group = X[labels == cluster, :]
                y_group = y[labels == cluster]

                if X_group.shape[0] <= 1:  # Prevent empty groups    WTF DOES THIS DO??
                    empty_group = True
                    group_sizes.append(0)
                    k_split_possible = False
                    break

                # Fit MARS model for this cluster
                sse, score, coeff, hinge_count, interaction, rearth = self.slope_.FitSpline(X_group, y_group)
                mars_models[cluster] = rearth

                hinge_counts.append(hinge_count)
                interactions.append(interaction)
                sse_values.append(sse)
                scores.append(score)
                group_sizes.append(len(y_group))

            if not k_split_possible:
                return 0, [], k_split_possible, 0

            # Reassign each data point to the best cluster based on residuals
            for j in range(X.shape[0]):
                x = X[j, :]
                y_actual = y[j]

                # Compute residuals for each cluster
                residuals_c = {}
                for c in range(k):
                    if c in mars_models:  # Check if the model exists for this cluster
                        residuals_c[c] = abs(y_actual - rf.predict_mars(x, mars_models[c]))
                    else:
                        # Handle the case where the model doesn't exist
                        residuals_c[c] = float('inf')  # Assign infinity or a large value to discourage assignment

                # Assign the point to the cluster with the smallest residual
                labels[j] = min(residuals_c, key=residuals_c.get)

            '''# Plot results if needed             Move to the other function maybe
            if needed_nodes:
                if len(pa_i) == 1:
                    self.plot.plot_2d_other(pa_i, variable_index, last_groups, "Method Iterations")
                elif len(pa_i) == 2:
                    self.plot.plot_3d_other(pa_i, variable_index, last_groups, "Method Iterations")
'''
            # Stopping criteria
            change_threshold = 0.005
            changes = np.sum(labels != last_groups)

            if (not first_iter and ((changes / len(labels)) < change_threshold)) or num_iter > 100:
                final_labels = labels.copy()
                print(f'Threshold BREAK: {changes / len(labels)} !!!')
                break

            if mdl_th:
                # Compute MDL Score for k Models
                cur_cost_split = self.ComputeScoreSplit(
                    hinge_counts, interactions, sse_values, scores, group_sizes,
                    self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False
                )
                if cur_cost_split >= prev_cost_split:
                    final_labels = labels.copy()
                    print(f'MDL Score BREAK: {cur_cost_split - prev_cost_split} !!!')
                    break
                prev_cost_split = cur_cost_split

            first_iter = False
            last_groups = labels.copy()

        cost_split = math.inf

        # Initialize lists to store model information for all clusters
        sse_list = []
        score_list = []
        hinge_counts_list = []
        interactions_list = []
        group_sizes = []
        mars_models = {}

        # Iterate over all clusters
        for cluster in range(k):
            # Get the data for the current cluster
            X_group = X[final_labels == cluster, :]
            y_group = y[final_labels == cluster]

            # Check if the group is not empty
            if X_group.shape[0] > 1:
                # Fit MARS model for this cluster
                sse, score, coeff, hinge_count, interactions, rearth = self.slope_.FitSpline(X_group, y_group)
                #y_pred = rf.predict_mars(X_group, rearth)

                # Store model information
                sse_list.append(sse)
                score_list.append(score)
                hinge_counts_list.append(hinge_count)
                interactions_list.append(interactions)
                group_sizes.append(len(y_group))
                mars_models[cluster] = rearth
            '''else:
                # Handle empty groups (e.g., assign default values) CHANGE THIS APPROACH
                sse_list.append(float('inf'))  # Large SSE to discourage selection
                score_list.append(0)
                hinge_counts_list.append(0)
                interactions_list.append(0)
                group_sizes.append(0)'''

        # Compute the score for all models
        cost_split = self.ComputeScoreSplit(
            hinge_counts_list, interactions_list, sse_list, score_list,
            group_sizes, self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False
        )

        return cost_split, final_labels, k_split_possible, num_iter

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
            total_model_cost += model_cost
            # Calculate residuals cost for the current cluster
            residuals_cost = self.slope_.gaussian_score_emp_sse(sse[i], rows[i], mindiff)
            total_residuals_cost += residuals_cost
            total_rows += rows[i]
        print('Total ROWS: ' + str(total_rows)) # Print total rows for debugging
        total_cost = base_cost + total_residuals_cost + total_model_cost + total_rows
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





