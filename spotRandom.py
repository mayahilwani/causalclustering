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


class SpotRandom:

    def __init__(self, max_int, log_results=True, vrb=True, dims=0):
        self.slope_ = Slope();
        self.vars = np.zeros((5, 5));
        self.gt = np.zeros((5, 5));
        self.attributes = [];
        self.M = max_int;
        self.log_path = "./logs/log_" + str(datetime.now(tz=None)).replace(' ', '_') + ".txt";
        self.log_flag = log_results;
        self.verbose = vrb;
        self.filename = "";
        self.result = None;
        self.split = np.zeros(10);
        self.scoref = np.zeros(10); # score full
        self.scores = np.zeros(10); # score split
        self.Nodes = [];
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.ordering = np.zeros(10)
        self.foundIntv = []
        self.intv = []
        #self.M = M;
        if self.log_flag:
            print("Saving results to: ", self.log_path)


    def loadData(self, filename):
        try:
            self.filename = filename;
            gt_file = f"{self.filename}/truth1.txt"
            gt = np.loadtxt(gt_file, delimiter=',')
            data_file1 = f"{self.filename}/data1.txt"
            #data_file2 = f"{self.filename}/dataintv1.txt"
            data_file3 = f"{self.filename}/interventions1.txt"
            data1 = np.loadtxt(data_file1, delimiter=',')
            #data2 = np.loadtxt(data_file2, delimiter=',')
            intvs = np.loadtxt(data_file3, delimiter=',', dtype=int)
            try:
                intvs = np.loadtxt(data_file3, delimiter=',', dtype=int)
            except ValueError as e:
                #print(f"Error: File empty: {e}")
                intvs = []  # Or handle as needed
            #if data1.shape[1] != data2.shape[1]:
            #    raise ValueError("The two files must have the same number of columns for vertical concatenation.")
            #variables = np.vstack((data1, data2))
            variables = data1
            attributes_file = f"{self.filename}/attributes1.txt"
            with open(attributes_file, "r") as atts:
                lines = atts.readlines()
                values = lines[1].strip()  # Second line contains the values
                # Convert the values to a list (optional)
                attributes = values.split(", ")

        except Exception as e:
            print(f"An error occurred: {e}")
        self.attributes = attributes;
        self.vars = variables;
        self.gt = gt;
        self.intv = intvs

    def run(self, needed_nodes = []):
        if not needed_nodes:
            nodestats_file = f"{self.filename}/node_STATS_rand.txt"
            with open(nodestats_file, "w") as stats:
                stats.write("id, num_parents, true_split, found_split, gmm_bic, score_diff, true_score_diff, num_iter, method_acc, gmm_acc, gmm_acc_res, kmeans_acc, kmeans_acc_res, f1, gmm_f1, gmm_f1_res, kmeans_f1, kmeans_f1_res\n")

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
            accuracy = 0
            f1 = 0
            gmm_f1 = 0
            gmm_f1_res = 0
            labels_xy = []
            kmeans_f1 = 0
            kmeans_f1_res = 0
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


            # Fit a Gaussian Mixture Model with 1 components
            gmm1 = GaussianMixture(n_components=1, random_state=19)
            gmm1.fit(residuals.reshape(-1, 1))

            # Fit a Gaussian Mixture Model with 2 components
            gmm2 = GaussianMixture(n_components=2, random_state=19)
            gmm2.fit(residuals.reshape(-1, 1))
            # Predict the cluster for each point
            #labels = gmm2.predict(residuals.reshape(-1, 1))
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
            #    print(
            #        f"Model with 2 components is preferred (BIC: {bic2:.2f}) over model with 1 component (BIC: {bic1:.2f}).")

            # Split the data into two groups based on the labels
            #group1 = residuals[labels == 0]
            #group2 = residuals[labels == 1]
            final_labels = labels.copy()
            first_iter = True
            num_iter = 0
            last_groups = labels.copy()
            only_one = False
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
                residuals_group1 = y_group1 - y_pred1
                # Group2 MARS
                sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2,
                                                                                                       y_group2)
                y_pred2 = rf.predict_mars(X_group2, rearth2)
                residuals_group2 = y_group2 - y_pred2
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

                # if num_iter == 5:
                change_threshold = 0.015 #0.05  # e.g., require less than 1% of changes to stop
                changes = np.sum(labels != last_groups)
                if (not first_iter and (changes / len(labels)) < change_threshold) or num_iter > 100: # FOR NOW
                    final_labels = labels.copy()
                    print('threshold BREAK ' + str(changes / len(labels)) + ' !!!')
                    #print('Initial split is settled for node ' + str(variable_index) + ' !!!')
                    break

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
                cost_split = self.ComputeScoreSplit(hinge_count1, interactions1, sse1, score1, rows1, hinge_count2,
                                                     interactions2, sse2, score2, rows2, self.Nodes[i].min_diff,
                                                     np.array([len(pa_i)]), show_graph=False)


            if needed_nodes:
                # PLOTING THE FINAL RESULT IF ONLY 1 PARENT
                if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
                    if not only_one:
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
                    X_all = self.vars[:, pa_i]
                    y_all = self.vars[:, variable_index]
                    sorted_indices_all = np.argsort(X_all.flatten())
                    X_all_sorted = X_all[sorted_indices_all]
                    y_all_sorted = y_pred[sorted_indices_all]
                    ax.plot(X_all_sorted, y_all_sorted, color='green', linestyle='-', linewidth=2,
                            label='MARS (All Data)')
                    if only_one:
                        ax.scatter(X_all, y_all, color='purple', alpha=0.5, label='Group 1')


                    ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
                    X_orig = X[:int(self.attributes[2])]
                    y_orig = y[:int(self.attributes[2])]
                    X_intv = X[int(self.attributes[2]):]
                    y_intv = y[int(self.attributes[2]):]
                    ax.scatter(X_orig, y_orig, color='blue', alpha=0.5, label='Original')
                    ax.scatter(X_intv, y_intv, color='red', alpha=0.5, label='Intervention')
                    ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

                    # New: KMeans Clustering Plot
                    fig, ax = plt.subplots(figsize=(7, 7))
                    scatter = ax.scatter(X_all, y_all, c=kmeans_labels_xy, cmap='viridis', alpha=0.5)
                    ax.set_title(f'KMeans Clusters for Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    plt.tight_layout()
                    plt.show()

                    # New: GMM Clustering Plot
                    fig, ax = plt.subplots(figsize=(7, 7))
                    scatter = ax.scatter(X_all, y_all, c=labels_xy, cmap='viridis', alpha=0.5)
                    ax.set_title(f'GMM Clusters for Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    plt.tight_layout()
                    plt.show()

                # PLOT THE FINAL RESULT IF 2 PARENTS
                if len(pa_i) == 2:  # Only plot when there are two parents (3D case)
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    # Extract parent variables
                    X_parent1 = self.vars[:, pa_i[0]]
                    X_parent2 = self.vars[:, pa_i[1]]
                    y_all = self.vars[:, variable_index]
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
                        X1_mesh.shape)  # Replace `mars_model` with your prediction model

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

                    # New: KMeans 3D Clustering Plot
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(X_parent1, X_parent2, y_all, c=kmeans_labels_xy, cmap='viridis', alpha=0.5)
                    ax.set_title(f'KMeans Clusters for Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
                    ax.set_xlabel(f'Parent Variable {pa_i[0]}')
                    ax.set_ylabel(f'Parent Variable {pa_i[1]}')
                    ax.set_zlabel(f'Variable {variable_index}')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    plt.tight_layout()
                    plt.show()

                    # New: GMM 3D Clustering Plot
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(X_parent1, X_parent2, y_all, c=labels_xy, cmap='viridis', alpha=0.5)
                    ax.set_title(f'GMM Clusters for Variable {variable_index} vs Parents {pa_i[0]} and {pa_i[1]}')
                    ax.set_xlabel(f'Parent Variable {pa_i[0]}')
                    ax.set_ylabel(f'Parent Variable {pa_i[1]}')
                    ax.set_zlabel(f'Variable {variable_index}')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    plt.tight_layout()
                    plt.show()


            # Score for ONE MODEL
            cost_all = self.ComputeScore(hinge_count, interactions, sse, score, len(y), self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)
            print('COST for initial model is ' + str(cost_all))
            final_cost = cost_all

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

                    # Check if it is one of the intv nodes if not:

                else:
                    #self.result = 0
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

                true_cost_split = self.ComputeScoreSplit(hinge_count1, interactions1, sse1, score1, rows1, hinge_count2,
                                                         interactions2, sse2, score2, rows2, self.Nodes[i].min_diff,
                                                         np.array([len(pa_i)]), show_graph=False)
                true_cost_gain = cost_all - true_cost_split

            self.scoref[variable_index] = cost_all
            self.scores[variable_index] = final_cost
            self.result[variable_index] = final_labels
            score_diff = cost_all - cost_split
            if not needed_nodes:
                # id, num_parents, true_split, found_split, score_diff
                with open(nodestats_file, "a") as stats:
                    stats.write(f"{variable_index},{len(pa_i)}, {is_intv},{is_intv_found},{gmm_bic},{score_diff},{true_cost_gain},{num_iter},{accuracy},{gmm_acc},{gmm_acc_res},{kmeans_acc},{kmeans_acc_res},{f1},{gmm_f1},{gmm_f1_res},{kmeans_f1},{kmeans_f1_res} \n")

        return self.foundIntv, accuracies
        # Finalize the logger
        #logger.WriteLog("END LOGGING FOR FILE: " + self.filename)
        #logger.End()

    def compute_f1(self, predicted_labels, labels_true):
        """Compute the best F1-score considering both direct and switched alignments."""
        # Case 1: Direct alignment
        f1_case1 = f1_score(labels_true, predicted_labels)

        # Case 2: Switched alignment
        f1_case2 = f1_score(labels_true, 1 - predicted_labels)  # Flip labels

        return max(f1_case1, f1_case2)

    # ComputeScore(source,target,rows,child.GetMinDiff(),k=np.array([1]))
    def ComputeScore(self, hinges, interactions, sse, model, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        #print('BASE-cost: ' + str(base_cost))
        model_cost = self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        #print('Model :  ' + str(model))
        residuals_cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff)
        residuals_avg_cost = residuals_cost/rows
        #print('ROWS  is: ' + str(rows))
        #print('AVG Cost of the residuals is (ONE): ' + str(residuals_avg_cost))
        models = model + model_cost
        #print('ONE Model cost: ' + str(models))
        #cost = residuals_avg_cost + base_cost + models/rows;
        cost = (residuals_cost + base_cost + models)
        #print('FULL COST of one model: ' + str(cost))
        #avg_cost = cost/ rows
        #print('AVG COST of 1 model: ' + str(avg_cost))
        return cost;

    def ComputeScoreSplit(self, hinges1, interactions1, sse1, model1, rows1, hinges2, interactions2, sse2, model2, rows2, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        #print('BASE-cost: ' + str(base_cost))
        model_cost1 = self.slope_.model_score(hinges1) + self.AggregateHinges(interactions1, k);
        model_cost2 = self.slope_.model_score(hinges2) + self.AggregateHinges(interactions2, k);
        #print('Model1 :  ' + str(model1))
        #print('Model2 :  ' + str(model2))
        #print('Cost of models ' + str(model_cost1) + ' and ' + str(model_cost2))
        # Calculate Gaussian scores for each group using their SSE and row count
        cost1 = self.slope_.gaussian_score_emp_sse(sse1, rows1, mindiff)
        cost2 = self.slope_.gaussian_score_emp_sse(sse2, rows2, mindiff)
        #residuals_avg_cost1 = cost1/rows1
        #residuals_avg_cost2 = cost2/rows2
        #residuals_avg_cost = ( residuals_avg_cost1 + residuals_avg_cost2)/2
        residuals_avg_cost = (cost1 + cost2) / (rows1 + rows2)
        print('ROWS1 and ROWS2: ' + str(rows1) + ', ' + str(rows2))
        #print('AVG Cost of the residuals is (TWO): ' + str(residuals_avg_cost))  #) + ' and ' + str(cost2))
        #print('Cost of the residuals1 is: ' + str(residuals_avg_cost1))
        #print('Cost of the residuals2 is: ' + str(residuals_avg_cost2))
        models = model1 + model2 + model_cost1 + model_cost2
        #print('TWO Model cost: ' + str(models))
        total_cost = base_cost + cost1 + cost2 + models + rows1 + rows2
        #print('FULL COST of two models: ' + str(total_cost))
        #model1_cost = (base_cost + model1 + model_cost1 + cost1) / rows1
        #model2_cost = (base_cost + model2 + model_cost2 + cost2) / rows2
        #print('COST of M1: ' + str(model1_cost))
        #print('COST of M2: ' + str(model2_cost))
        #avg_total_cost = (model1_cost + model2_cost) / 2
        #avg_total_cost = total_cost / (rows1 + rows2)
        #print('Avg COST of 2: ' + str(avg_total_cost))
        #total_cost = residuals_avg_cost + base_cost + models/(rows1 + rows2)
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

