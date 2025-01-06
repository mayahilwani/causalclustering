from node import Node;
from edge import Edge;
from slope import Slope;
from globe import Globe;
from utils import *
#from dag import DAG;
from logger import Logger
import numpy as np;
import sys
import time as time
import os;
import gc;
from datetime import datetime
from queue import PriorityQueue
from top_sort import *
import RFunctions as rf
from sklearn.mixture import GaussianMixture
import seaborn as sns
from combinator import Combinator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GlobeWrapper:

    def __init__(self, max_int, log_results=True, vrb=True, dims=0, M=2):
        self.slope_ = Slope();
        self.vars = np.zeros((5, 5));
        self.gt = np.zeros((5, 5));
        #self.M = max_int;
        self.log_path = "./logs/log_" + str(datetime.now(tz=None)).replace(' ', '_') + ".txt";
        self.log_flag = log_results;
        self.verbose = vrb;
        self.filename = "";
        self.result = 0;
        self.Nodes = [];
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.M = M;
        if self.log_flag:
            print("Saving results to: ", self.log_path)


    def loadData(self, filename):
        try:
            base_path = "C:/Users/ziadh/Documents/CausalGen-Osman/maya_data"
            self.filename = base_path + "/" + filename;
            gt_file = f"{self.filename}/truth1.txt"
            gt = np.loadtxt(gt_file, delimiter=',')
            data_file1 = f"{self.filename}/data1.txt"
            data_file2 = f"{self.filename}/dataintv1.txt"
            data1 = np.loadtxt(data_file1, delimiter=',')
            data2 = np.loadtxt(data_file2, delimiter=',')
            if data1.shape[1] != data2.shape[1]:
                raise ValueError("The two files must have the same number of columns for vertical concatenation.")
            data = np.vstack((data1, data2))
            print('DATA SHAPE: ' + str(data.shape))
            #plot_residuals(gt, data)
            variables = data

        except Exception as e:
            print(f"An error occurred: {e}")

        self.vars = variables;
        self.gt = gt;

    def run(self):
        scores = []
        # Standardize the loaded data (already loaded in self.vars via loadData)
        print('VARS BEFORE:  ' + str(self.vars.shape ))
        normalized_vars = Standardize(self.vars)
        print('VARS AFTER:  ' + str(normalized_vars.shape))
        recs = normalized_vars.shape[0]
        dim = normalized_vars.shape[1]
        self.V = dim
        headers = [i for i in range(0, dim)]

        # Initialize Slope and Globe objects
        globe_ = Globe(self.slope_, dims=dim, M=self.M)

        # Initialize the logger
        #logger = Logger(self.log_path, log_to_disk=self.log_flag, verbose=self.verbose)
        #logger.Begin()
        #logger.WriteLog("BEGIN LOGGING FOR FILE: " + self.filename)

        dims = self.gt.shape[1]
        g = Graph(dims)
        # Load the ground truth network from gt (already loaded in self.vars in loadData)
        gt_network = self.gt

        # Create Edges based on ground truth adjacency matrix
        Edges = [[None for _ in range(dim)] for _ in range(dim)]

        # Construct the Edges from the ground truth (gt_network)
        for i in range(dims):
            for j in range(dims):
                if gt_network[i, j] == 1:
                    g.addEdge(i, j)
                    Edges[i][j] = Edge(i, j, [], 0)
        ordering = g.nonRecursiveTopologicalSort()

        # Create Node objects from normalized variables
        for node in ordering:
            self.Nodes.append(Node(normalized_vars[:, node].reshape(recs, -1), globe_))
        print('NODES HAS:  ' + str(self.Nodes) + '  ORDERINGS HAS: ' + str(ordering))

        # Traverse the nodes top to bottom
        for i,variable_index in enumerate(ordering):
            # Get parents of the node
            print('i VALUE is: ' + str(i) + ' and variable index: ' + str(variable_index))
            pa_i = np.where(gt_network[:, variable_index] == 1)[0]
            print(f"Variable {variable_index} has parents {pa_i}")

            # If node has no parents print and skip it
            if len(pa_i) == 0:
                print(f"Variable {variable_index} has no parents.")
                continue

            X = self.vars[:, pa_i]
            y = self.vars[:, variable_index]

            # Fit MARS on the entire dataset
            sse,score,coeff,hinge_count,interactions, rearth = self.slope_.FitSpline(X,y)
            y_pred = rf.predict_mars(X, rearth)
            residuals = y - y_pred

            # Fit a Gaussian Mixture Model with 2 components
            gmm = GaussianMixture(n_components=2)
            gmm.fit(residuals.reshape(-1, 1))

            # Predict the cluster for each point
            labels = gmm.predict(residuals.reshape(-1, 1))

            # Split the data into two groups based on the labels
            group1 = residuals[labels == 0]
            group2 = residuals[labels == 1]
            last_groups = labels.copy()
            first_iter = True
            num_iter = 0
            while (True):
                num_iter = num_iter + 1
                # Fit different MARS lines for each group and check residuals
                groups = labels.reshape(-1, 1)
                X_group1 = X[labels == 0, :]
                y_group1 = y[labels == 0]
                X_group2 = X[labels == 1, :]
                y_group2 = y[labels == 1]

                # GROUP1 MARS
                sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = self.slope_.FitSpline(X_group1, y_group1)
                y_pred1 = rf.predict_mars(X_group1, rearth1)
                residuals_group1 = y_group1 - y_pred1
                # Group2 MARS
                sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = self.slope_.FitSpline(X_group2, y_group2)
                y_pred2 = rf.predict_mars(X_group2, rearth2)
                residuals_group2 = y_group2 - y_pred2

                #residuals_group1 = y_group1 - y_pred1
                #residuals_group2 = y_group2 - y_pred2

                # Plot residuals KDE for the full dataset
                #sns.kdeplot(residuals, fill=True)
                #plt.show()

                '''# Create subplots for side-by-side plots
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Plot the GMM clustering result
                axes[0].hist(group1, bins=30, alpha=0.5, label='Group 1', density=True)
                axes[0].hist(group2, bins=30, alpha=0.5, label='Group 2', density=True)
                axes[0].legend()
                axes[0].set_title('GMM Clustering Result for node ' + str(variable_index))
                axes[0].set_xlabel('Data values')
                axes[0].set_ylabel('Density')

                # Plot the KDE of the new residuals for each group
                sns.kdeplot(residuals_group1.flatten(), fill=True, ax=axes[1], label='Group 1 Residuals')
                sns.kdeplot(residuals_group2.flatten(), fill=True, ax=axes[1], label='Group 2 Residuals')
                axes[1].legend()
                axes[1].set_title('KDE of New Residuals for Each Group')
                axes[1].set_xlabel('Residual values')
                axes[1].set_ylabel('Density')

                # Display the plots
                plt.tight_layout()
                plt.show()'''

                if len(pa_i) == 1:  # Only plot when there's a single parent (2D case)
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
                    ax.scatter(X_group1, y_group1, color='blue', alpha=0.5, label='Group 1')
                    ax.scatter(X_group2, y_group2, color='red', alpha=0.5, label='Group 2')
                    # Plot the MARS line for the full dataset
                    X_all = self.vars[:, pa_i]
                    y_all = self.vars[:, variable_index]
                    sorted_indices_all = np.argsort(X_all.flatten())
                    X_all_sorted = X_all[sorted_indices_all]
                    y_all_sorted = y_pred[sorted_indices_all]
                    ax.plot(X_all_sorted, y_all_sorted, color='green', linestyle='-', linewidth=2,
                            label='MARS (All Data)')

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

                    ax.set_title(f'Variable {variable_index} vs Parent {pa_i}')
                    ax.set_xlabel(f'Parent Variable {pa_i}')
                    ax.set_ylabel(f'Variable {variable_index}')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

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

                if num_iter == 5:
                    last_groups = labels.copy()
                    print(f'Classification is settled for node ' + str(variable_index) + ' !!!')
                    break

                if first_iter:
                    first_iter = False

                last_groups = labels.copy()

            parents = []
            for j in pa_i:
                parents.append(self.Nodes[j])

            print('LENGTH OF NODES: ' + str(len(self.Nodes)) + ' and i is: ' + str(i))

            # sse,score,coeff,hinge_count,interactions, rearth = slope_.FitSpline(X,y)
            score_all = self.ComputeScore(hinge_count, interactions, sse, score, len(y), self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)
            print('SCORE for initial model is ' + str(score_all))

            rows2 = sum(last_groups)
            rows1 = len(last_groups) - rows2
            print(str(rows1) + ' members in group 1  and ' + str(rows2) + ' members in group 2')
            score_split = self.ComputeScoreSplit(hinge_count1, interactions1, sse1, score1, rows1, hinge_count2, interactions2, sse2, score2, rows2, self.Nodes[i].min_diff, np.array([len(pa_i)]), show_graph=False)
            print('SCORE for splitting model is ' + str(score_split))

            if score_all > score_split:
                self.result = 1
                print(f"Splitting model is better with score  {score_all-score_split}")
            else:
                self.result = 0
                print(f"Original model is better with score {score_split-score_all}")

        # Finalize the logger
        #logger.WriteLog("END LOGGING FOR FILE: " + self.filename)
        #logger.End()

    def ComputeScore(self, hinges, interactions, sse, model, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        print('BASE-cost: ' + str(base_cost))
        model_cost = self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        print('Model-cost: ' + str(model_cost))
        residuals_cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff)
        cost = residuals_cost + model + base_cost + model_cost;
        print('COST: ' + str(cost))
        return cost;

    def ComputeScoreSplit(self, hinges1, interactions1, sse1, model1, rows1, hinges2, interactions2, sse2, model2, rows2, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        print('BASE-cost: ' + str(base_cost))
        model_cost = self.slope_.model_score(hinges1) + self.slope_.model_score(hinges2) + self.AggregateHinges(interactions1, k) + self.AggregateHinges(interactions2, k);
        print('Cost of models ' + str(model_cost))
        # Calculate Gaussian scores for each group using their SSE and row count
        cost1 = self.slope_.gaussian_score_emp_sse(sse1, rows1, mindiff)
        cost2 = self.slope_.gaussian_score_emp_sse(sse2, rows2, mindiff)
        print('Cost of the residuals is: ' + str(cost1) + ' and ' + str(cost2))
        dp_assignments = 0#(rows1 + rows2) * 1  # Depending on how many clusters. We only have 2 for now.
        # Total cost is the sum of individual costs and the model contributions
        total_cost = cost1 + cost2 + model1 + model2 + base_cost + dp_assignments + model_cost
        return total_cost #, combined_coeffs;


    def AggregateHinges(self, hinges, k):
        cost = 0;
        flag = 1;

        for M in hinges:
            cost += self.slope_.logN(M) + Combinator(M, k) + M * np.log2(self.F);
        return cost;

