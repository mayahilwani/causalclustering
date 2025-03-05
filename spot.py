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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import adjusted_mutual_info_score
import math
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import SpectralClustering
from plotting import Plotting


class Spot:

    def __init__(self, max_int, log_results=True, vrb=True, dims=0):
        self.M = max_int;
        self.log_flag = log_results;
        self.verbose = vrb;
        self.V = dims;

        self.slope_ = Slope();
        self.plot = None
        self.vars = np.zeros((5, 5));
        self.gt = np.zeros((5, 5));
        self.attributes = [];
        self.log_path = "./logs/log_" + str(datetime.now(tz=None)).replace(' ', '_') + ".txt";
        self.filename = "";
        self.result = None;
        self.split = np.zeros(10);
        self.scoref = np.zeros(10); # score full
        self.scores = np.zeros(10); # score split
        self.Nodes = [];
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.ordering = np.zeros(10)
        self.foundIntv = []
        self.intv = []
        if self.log_flag:
            print("Saving results to: ", self.log_path)


    def loadData(self, filename):
        try:
            self.filename = filename;
            gt_file = f"{self.filename}/truth1.txt"
            gt = np.loadtxt(gt_file, delimiter=',')
            data_file1 = f"{self.filename}/data1.txt"
            data_file3 = f"{self.filename}/interventions1.txt"
            data1 = np.loadtxt(data_file1, delimiter=',')
            intvs = np.loadtxt(data_file3, delimiter=',', dtype=int)
            try:
                intvs = np.loadtxt(data_file3, delimiter=',', dtype=int)
            except ValueError as e:
                intvs = []  # Or handle as needed
            variables = data1
            attributes_file = f"{self.filename}/attributes1.txt"
            with open(attributes_file, "r") as atts:
                lines = atts.readlines()
                values = lines[1].strip()  # Second line contains the values
                # Convert the values to a list
                attributes = values.split(", ")
        except Exception as e:
            print(f"An error occurred: {e}")
        self.attributes = attributes
        self.vars = variables
        self.gt = gt
        self.intv = intvs
        self.plot = Plotting(variables)

    def run(self, needed_nodes = [], random = False):
        if not needed_nodes:
            nodestats_file = f"{self.filename}/node_STATS.txt"
            if random:
                nodestats_file = f"{self.filename}/node_STATS_rand.txt"
            with open(nodestats_file, "w") as stats:
                stats.write("id, num_parents, true_split, found_split, gmm_bic, score_diff, true_score_diff, num_iter, method_acc, gmm_acc, gmm_acc_res, kmeans_acc, kmeans_acc_res, spectral_acc, spectral_acc_res, f1, gmm_f1, gmm_f1_res, kmeans_f1, kmeans_f1_res, spectral_f1, spectral_f1_res\n")

        # Standardize the loaded data (already loaded in self.vars via loadData)
        normalized_vars = self.vars  #Standardize(self.vars)
        #normalized_vars = self.vars
        recs = normalized_vars.shape[0]
        dim = normalized_vars.shape[1]
        self.V = dim
        self.result = np.zeros((dim, recs))
        headers = [i for i in range(0, dim)]

        # Initialize the logger
        #logger = Logger(self.log_path, log_to_disk=self.log_flag, verbose=self.verbose)
        #logger.Begin()
        #logger.WriteLog("BEGIN LOGGING FOR FILE: " + self.filename)

        dims = self.gt.shape[1]
        g = Graph(dims)
        # Load the ground truth network from gt
        gt_network = self.gt

        if not needed_nodes:
            # Create Edges based on ground truth adjacency matrix
            Edges = [[None for _ in range(dim)] for _ in range(dim)]

            # Construct the Edges from the ground truth (gt_network)
            for i in range(dims):
                for j in range(dims):
                    if gt_network[i, j] == 1:
                        g.addEdge(i, j)
                        Edges[i][j] = Edge(i, j, [], 0)
            ordering = g.nonRecursiveTopologicalSort()
            self.ordering = ordering
        else:
            self.ordering = needed_nodes
        print('ORDERING ' + str(self.ordering))
        # Create Node objects from normalized variables
        for node in range(0, dim):
            self.Nodes.append(Node(normalized_vars[:, node].reshape(recs, -1), self))

        accuracies = []
        print(f"Interventions {self.intv}.")
        # Traverse the nodes top to bottom
        for i,variable_index in enumerate(self.ordering):
            gmm_acc = 0
            kmeans_acc = 0
            gmm_acc_res = 0
            kmeans_acc_res = 0
            spectral_acc = 0
            spectral_acc_res = 0
            accuracy = 0
            f1 = 0
            gmm_f1 = 0
            gmm_f1_res = 0
            labels_xy = []
            kmeans_f1 = 0
            kmeans_f1_res = 0
            spectral_f1 = 0
            spectral_f1_res = 0
            kmeans_labels_xy = []
            is_intv_found = 0
            gmm_bic = 0
            is_intv =  1 if variable_index in self.intv else 0

            # Get parents of the node
            print('Iteration: ' + str(i))
            pa_i = np.where(gt_network[:, variable_index] == 1)[0]
            print(f"NODE {variable_index} has parents {pa_i}")

            # If node has no parents print and skip it
            if len(pa_i) == 0:
                print(f"Node {variable_index} has no parents.")
                continue

            X = self.vars[:, pa_i]
            y = self.vars[:, variable_index]
            data_xy = np.hstack((X, y.reshape(-1, 1)))  # Combine X and y

            # Fit MARS on the entire dataset
            sse,score,coeff,hinge_count,interactions, rearth = self.slope_.FitSpline(X,y)
            y_pred = rf.predict_mars(X, rearth)
            residuals = y - y_pred

            # Score for ONE MODEL
            cost_all = self.ComputeScore(hinge_count, interactions, sse, score, len(y), self.Nodes[i].min_diff,
                                         np.array([len(pa_i)]), show_graph=False)
            print('COST for initial model is ' + str(cost_all))
            final_cost = cost_all

            if is_intv:
                labels_true = np.array([0] * int(self.attributes[2]) + [1] * int(self.attributes[3]))
                ### Compute GMM Accuracy on (X, y) ###
                gmm_xy = GaussianMixture(n_components=2, random_state=19)
                gmm_xy.fit(data_xy)
                labels_xy = gmm_xy.predict(data_xy)

                gmm_acc_case1 = np.mean((labels_xy == 1) == labels_true)
                gmm_acc_case2 = np.mean((labels_xy == 0) == labels_true)
                gmm_acc = max(gmm_acc_case1, gmm_acc_case2)

                gmm_f1 = self.compute_f1(labels_xy, labels_true)

                ### Compute GMM Accuracy on Residuals ###
                gmm_res = GaussianMixture(n_components=2, random_state=19)
                gmm_res.fit(residuals.reshape(-1, 1))
                labels_res = gmm_res.predict(residuals.reshape(-1, 1))

                gmm_acc_res_case1 = np.mean((labels_res == 1) == labels_true)
                gmm_acc_res_case2 = np.mean((labels_res == 0) == labels_true)
                gmm_acc_res = max(gmm_acc_res_case1, gmm_acc_res_case2)

                gmm_f1_res = self.compute_f1(labels_res, labels_true)

                ### Compute K-Means Accuracy on (X, y) ###
                kmeans_xy = KMeans(n_clusters=2, random_state=19)
                kmeans_xy.fit(data_xy)
                kmeans_labels_xy = kmeans_xy.labels_

                kmeans_acc_case1 = np.mean((kmeans_labels_xy == 1) == labels_true)
                kmeans_acc_case2 = np.mean((kmeans_labels_xy == 0) == labels_true)
                kmeans_acc = max(kmeans_acc_case1, kmeans_acc_case2)

                kmeans_f1 = self.compute_f1(kmeans_labels_xy, labels_true)

                ### Compute K-Means Accuracy on Residuals ###
                kmeans_res = KMeans(n_clusters=2, random_state=19)
                kmeans_res.fit(residuals.reshape(-1, 1))
                kmeans_labels_res = kmeans_res.labels_

                kmeans_acc_res_case1 = np.mean((kmeans_labels_res == 1) == labels_true)
                kmeans_acc_res_case2 = np.mean((kmeans_labels_res == 0) == labels_true)
                kmeans_acc_res = max(kmeans_acc_res_case1, kmeans_acc_res_case2)

                kmeans_f1_res = self.compute_f1(kmeans_labels_res, labels_true)

                ### Compute Spectral Clustering Accuracy on (X, y) ###
                spectral_xy = SpectralClustering(n_clusters=2, random_state=19)
                spectral_xy.fit(data_xy)
                spectral_labels_xy = spectral_xy.labels_

                spectral_acc_case1 = np.mean((spectral_labels_xy == 1) == labels_true)
                spectral_acc_case2 = np.mean((spectral_labels_xy == 0) == labels_true)
                spectral_acc = max(spectral_acc_case1, spectral_acc_case2)

                spectral_f1 = self.compute_f1(spectral_labels_xy, labels_true)

                ### Compute Spectral Clustering Accuracy on Residuals ###
                spectral_res = SpectralClustering(n_clusters=2, random_state=19)
                spectral_res.fit(residuals.reshape(-1, 1))
                spectral_labels_res = spectral_res.labels_

                spectral_acc_res_case1 = np.mean((spectral_labels_res == 1) == labels_true)
                spectral_acc_res_case2 = np.mean((spectral_labels_res == 0) == labels_true)
                spectral_acc_res = max(spectral_acc_res_case1, spectral_acc_res_case2)

                spectral_f1_res = self.compute_f1(spectral_labels_res, labels_true)


            # Fit a Gaussian Mixture Model with 1 components
            gmm1 = GaussianMixture(n_components=1, random_state=19)
            gmm1.fit(residuals.reshape(-1, 1))

            # Fit a Gaussian Mixture Model with 2 components
            gmm2 = GaussianMixture(n_components=2, random_state=19)
            gmm2.fit(residuals.reshape(-1, 1))
            # Predict the cluster for each point
            labels = gmm2.predict(residuals.reshape(-1, 1))

            if random:
                labels = np.random.choice([0, 1], size=residuals.shape[0])

            # gmm.bic get BIC score and compair it with n_components=1    <<<<<<<<<<<<<<<<<<<<<<<
            # Get BIC scores
            bic1 = gmm1.bic(residuals.reshape(-1, 1))
            bic2 = gmm2.bic(residuals.reshape(-1, 1))

            # Compare BIC scores
            if bic1 <= bic2:
                gmm_bic = 0
            #    print(
            #        f"Model with 1 component is preferred (BIC: {bic1:.2f}) over model with 2 components (BIC: {bic2:.2f}).")
            else:
                gmm_bic = 1

            final_labels = labels.copy()
            first_iter = True
            num_iter = 0
            last_groups = labels.copy()
            only_one = False
            prev_cost_split = math.inf
            while (True):
                num_iter = num_iter + 1
                # Fit different MARS lines for each group and check residuals
                groups = labels.reshape(-1, 1)
                X_group1 = X[labels == 0, :]
                y_group1 = y[labels == 0]
                X_group2 = X[labels == 1, :]
                y_group2 = y[labels == 1]

                if X_group1.shape[0] <= 1 or X_group2.shape[0] <= 1:
                    only_one = True
                    labels = np.zeros(labels.shape)
                    #print('LABELS AFTER only_one=True NOW HAS SUM: ' + str(sum(labels)))
                    final_labels = labels.copy()
                    break

                # GROUP1 MARS
                sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = self.slope_.FitSpline(X_group1,
                                                                                                   y_group1)
                y_pred1 = rf.predict_mars(X_group1, rearth1)
                #residuals_group1 = y_group1 - y_pred1
                # Group2 MARS
                sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2,
                                                                                                       y_group2)
                y_pred2 = rf.predict_mars(X_group2, rearth2)
                #residuals_group2 = y_group2 - y_pred2

                # MDL Score for TWO MODELS
                rows2 = sum(labels)  # == 0 ????
                rows1 = len(labels) - rows2
                #print(str(rows1) + ' members in group 1  and ' + str(rows2) + ' members in group 2')
                cur_cost_split = self.ComputeScoreSplit([hinge_count1, hinge_count2], [interactions1, interactions2], [sse1, sse2], [score1, score2],
                                                        [rows1, rows2], self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)

                # Reassign the data points
                for j in range(X.shape[0]):
                    x = X[j, :]
                    y_predict_1 = rf.predict_mars(x, rearth1)
                    y_predict_2 = rf.predict_mars(x, rearth2)
                    res1 = abs(y[j] - y_predict_1)
                    res2 = abs(y[j] - y_predict_2)
                    if res1 < res2:
                        labels[j] = 0
                    else:
                        labels[j] = 1
                if needed_nodes:
                    # PLOTING THE FINAL RESULT IF ONLY 1 PARENT
                    if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
                        self.plot.plot_2d_other(pa_i, variable_index, last_groups, "Method Iterations")
                    elif len(pa_i) == 2:
                        self.plot.plot_3d_other(pa_i, variable_index, last_groups, "Method Iterations")

                change_threshold = 0.005 #0.05  # e.g., require less than 1% of changes to stop
                changes = np.sum(labels != last_groups)
                if (not first_iter and ( (changes / len(labels)) < change_threshold ) ) or num_iter > 100: # FOR NOW
                    final_labels = labels.copy()
                    print('threshold BREAK ' + str(changes / len(labels)) + ' !!!')
                    #print('Initial split is settled for node ' + str(variable_index) + ' !!!')
                    break
                if cur_cost_split >= prev_cost_split:
                    final_labels = labels.copy()
                    print('MDL score BREAK ' + str(cur_cost_split - prev_cost_split) + ' !!!')
                    break

                prev_cost_split = cur_cost_split
                if first_iter:
                    first_iter = False
                last_groups = labels.copy()

            cost_split = math.inf

            # Fit two models when it is possible
            y_pred1 = y_pred
            y_pred2 = y_pred
            X_group1 = X[final_labels == 0, :]
            y_group1 = y[final_labels == 0]
            X_group2 = X[final_labels == 1, :]
            y_group2 = y[final_labels == 1]
            if X_group1.shape[0] <= 1 or X_group2.shape[0] <= 1:
                only_one = True
            if not only_one:
                # GROUP1 MARS
                sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = self.slope_.FitSpline(X_group1, y_group1)
                y_pred1 = rf.predict_mars(X_group1, rearth1)
                #residuals_group1 = y_group1 - y_pred1
                # Group2 MARS
                sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2, y_group2)
                y_pred2 = rf.predict_mars(X_group2, rearth2)
                #residuals_group2 = y_group2 - y_pred2

                # Score for TWO MODELS
                rows2 = sum(final_labels)  # == 0 ????
                rows1 = len(final_labels) - rows2
                print(str(rows1) + ' members in group 1  and ' + str(rows2) + ' members in group 2')
                cost_split = self.ComputeScoreSplit([hinge_count1, hinge_count2], [interactions1, interactions2], [sse1, sse2], [score1, score2],
                                                        [rows1, rows2], self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)


            if needed_nodes:
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
                    self.plot.plot_3d_other(pa_i, variable_index, spectral_labels_xy, "Spectral")

            print('COST for splitting model is ' + str(cost_split))
            eps = 0 #3100  # THREASHOLD FOR MDL DECISION (BITS)
            if not only_one:
                if cost_split < cost_all:
                    self.split[variable_index] = 1
                    is_intv_found = 1
                    print(f"Splitting model is better with score  {cost_all - cost_split}")
                    final_cost = cost_split
                    self.foundIntv.append(variable_index)
                    labels_true = np.array([0] * int(self.attributes[2]) + [1] * int(self.attributes[3]))
                    ami_score = adjusted_mutual_info_score(labels_true, final_labels)
                    print("Adjusted Mutual Information Score:", ami_score)

                    # Compute case-wise accuracy
                    accuracy_case1 = np.mean((final_labels == 1) == labels_true)
                    accuracy_case2 = np.mean((final_labels == 0) == labels_true)

                    # Choose the best case
                    if accuracy_case1 >= accuracy_case2:
                        TP = np.sum((final_labels == 1) & (labels_true == 1))  # True Positives
                        TN = np.sum((final_labels == 0) & (labels_true == 0))  # True Negatives
                        accuracy = accuracy_case1
                        final_pred_labels = final_labels  # Use this for F1-score computation
                    else:
                        TP = np.sum((final_labels == 0) & (labels_true == 1))  # True Positives (switched)
                        TN = np.sum((final_labels == 1) & (labels_true == 0))  # True Negatives (switched)
                        accuracy = accuracy_case2
                        final_pred_labels = 1 - final_labels
                    f1 = f1_score(labels_true, final_pred_labels)
                    print(f"Accuracy: {accuracy}")
                    accuracies.append(accuracy)

                    # Print the results
                    print(f"True Positives (TP): {TP}")
                    print(f"True Negatives (TN): {TN}")

                else:
                    print(f"Original model is better with cost difference {cost_split-cost_all}")
            else:
                print(f"Original model is better with cost {cost_all}  SPLITTING NOT POSSIBLE")
            true_cost_gain = 0
            if is_intv:
                X_group1 = X[labels_true == 0, :]
                y_group1 = y[labels_true == 0]
                X_group2 = X[labels_true == 1, :]
                y_group2 = y[labels_true == 1]
                # GROUP1 MARS
                sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = self.slope_.FitSpline(X_group1, y_group1)
                #y_pred1 = rf.predict_mars(X_group1, rearth1)
                # residuals_group1 = y_group1 - y_pred1
                # Group2 MARS
                sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2, y_group2)
                #y_pred2 = rf.predict_mars(X_group2, rearth2)
                # residuals_group2 = y_group2 - y_pred2
                # Score for TWO MODELS
                rows1 = int(self.attributes[2])
                rows2 = int(self.attributes[3])

                true_cost_split = self.ComputeScoreSplit([hinge_count1, hinge_count2], [interactions1, interactions2], [sse1, sse2], [score1, score2],
                                                        [rows1, rows2], self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)
                true_cost_gain = cost_all - true_cost_split

            self.scoref[variable_index] = cost_all
            self.scores[variable_index] = final_cost
            self.result[variable_index] = final_labels
            score_diff = cost_all - cost_split
            if not needed_nodes:
                # id, num_parents, true_split, found_split, score_diff
                with open(nodestats_file, "a") as stats:
                    stats.write(f"{variable_index},{len(pa_i)}, {is_intv},{is_intv_found},{gmm_bic},{score_diff},{true_cost_gain},{num_iter},{accuracy},{gmm_acc},{gmm_acc_res},{kmeans_acc},{kmeans_acc_res},{spectral_acc},{spectral_acc_res},{f1},{gmm_f1},{gmm_f1_res},{kmeans_f1},{kmeans_f1_res},{spectral_f1},{spectral_f1_res} \n")

        return self.foundIntv, accuracies
        # Finalize the logger
        #logger.WriteLog("END LOGGING FOR FILE: " + self.filename)
        #logger.End()

    # CLUSTER DAG ANALYSIS PART
    def compute_f1(self, predicted_labels, labels_true):
        """Compute the best F1-score considering both direct and switched alignments."""
        # Case 1: Direct alignment
        f1_case1 = f1_score(labels_true, predicted_labels)

        # Case 2: Switched alignment
        f1_case2 = f1_score(labels_true, 1 - predicted_labels)  # Flip labels

        return max(f1_case1, f1_case2)

    def analyzeLabels(self):
        for node in self.ordering:
            if self.split[node] == 1:
                pa_i = np.where(self.gt[:, node] == 1)[0]
                print('CHILD NODE IS: ' + str(node))
                for pa in pa_i:
                    pa_split = self.chooseBestSplit(node, pa)


    def chooseBestSplit(self, node, pa):
        child_labels = self.result[node]
        parent_labels = self.result[pa]
        score1 = self.applyLabels(pa, child_labels)
        if self.split[pa]:
            score2 = self.scores[pa]
            print('PARENT IS SPLIT')
        else:
            score2 = self.scoref[pa]
            print('PARENT IS NOT SPLIT')
        print('Score of' + str(pa) +' parent (original): '+ str(score2) + ' Score with child labels: '+ str(score1))
        if score1 <= score2:
            print( 'Child labels have a good fit. ')
            self.split[pa] = child_labels

        else:
            print('Original labels BETTER by ' + str(score1 - score2))


    def applyLabels(self, node, new_labels):
        pa_i = np.where(self.gt[:, node] == 1)[0]
        print(f"Variable {node} has parents {pa_i}")

        # If node has no parents print and skip it
        if len(pa_i) == 0:
            print(f"Variable {node} has no parents.")
            return 0
        X = self.vars[:, pa_i]
        y = self.vars[:, node]
        groups = new_labels.reshape(-1, 1)

        X_group1 = X[new_labels == 0, :]
        y_group1 = y[new_labels == 0]
        X_group2 = X[new_labels == 1, :]
        y_group2 = y[new_labels == 1]

        # GROUP1 MARS
        sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = self.slope_.FitSpline(X_group1, y_group1)
        y_pred1 = rf.predict_mars(X_group1, rearth1)
        residuals_group1 = y_group1 - y_pred1
        # Group2 MARS
        sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2, y_group2)
        y_pred2 = rf.predict_mars(X_group2, rearth2)
        residuals_group2 = y_group2 - y_pred2

        rows2 = sum(new_labels)
        rows1 = len(new_labels) - rows2
        print(str(rows1) + ' members in group 1  and ' + str(rows2) + ' members in group 2')
        score_split = self.ComputeScoreSplit(hinge_count1, interactions1, sse1, score1, rows1, hinge_count2,
                                             interactions2, sse2, score2, rows2, self.Nodes[node].min_diff,
                                             np.array([len(pa_i)]), show_graph=False)
        print('SCORE for splitting model is ' + str(score_split))
        return score_split

    # SCORE COMPUTATION PART
    def ComputeScore(self, hinges, interactions, sse, model, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        model_cost = self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        residuals_cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff)
        models = model + model_cost
        cost = (residuals_cost + base_cost + models)
        return cost;

    '''def ComputeScoreSplit(self, hinges1, interactions1, sse1, model1, rows1, hinges2, interactions2, sse2, model2, rows2, mindiff, m, show_graph=False):
        base_cost = self.slope_.model_score(m) + m * np.log2(self.V); # m is the number of parent variables
        model_cost1 = self.slope_.model_score(hinges1) + self.AggregateHinges(interactions1, m);
        model_cost2 = self.slope_.model_score(hinges2) + self.AggregateHinges(interactions2, m);
        cost1 = self.slope_.gaussian_score_emp_sse(sse1, rows1, mindiff)
        cost2 = self.slope_.gaussian_score_emp_sse(sse2, rows2, mindiff)
        residuals_avg_cost = (cost1 + cost2) / (rows1 + rows2)
        print('ROWS1 and ROWS2: ' + str(rows1) + ', ' + str(rows2))
        models = model1 + model2 + model_cost1 + model_cost2
        total_cost = base_cost + cost1 + cost2 + models + rows1 + rows2
        return total_cost'''

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

