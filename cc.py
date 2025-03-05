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
import math
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import SpectralClustering
from plotting import Plotting

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
        self.result = []
        self.split = [] #np.zeros(10);
        self.scoref = [] #np.zeros(10);  # score full
        self.scores = [] #np.zeros(10);  # score split
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

    def run(self, needed_nodes = [], random = False):
        if not needed_nodes:
            nodestats_file = f"{self.filename}/node_STATS.txt"
            if random:
                nodestats_file = f"{self.filename}/node_STATS_rand.txt"
            with open(nodestats_file, "w") as stats:
                stats.write("id, num_parents, true_split, found_split, gmm_bic, score_diff, true_score_diff, num_iter, method_acc, gmm_acc, gmm_acc_res, kmeans_acc, kmeans_acc_res, spectral_acc, spectral_acc_res, f1, gmm_f1, gmm_f1_res, kmeans_f1, kmeans_f1_res, spectral_f1, spectral_f1_res\n")

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
            cc_acc = 0
            gmm_acc = 0
            kmeans_acc = 0
            spectral_acc = 0
            gmm_acc_res = 0
            kmeans_acc_res = 0
            spectral_acc_res = 0
            cc_f1 = 0
            gmm_f1 = 0
            kmeans_f1 = 0
            spectral_f1 = 0
            gmm_f1_res = 0
            kmeans_f1_res = 0
            spectral_f1_res = 0
            #gmm_labels_xy = []
            #kmeans_labels_xy = []
            is_intv_found = 0
            gmm_bic = 0
            is_intv =  1 if variable_index in self.intv else 0

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

            if is_intv:
                gmm_acc, kmeans_acc, spectral_acc, gmm_acc_res, kmeans_acc_res, spectral_acc_res, gmm_f1, kmeans_f1, spectral_f1, gmm_f1_res, kmeans_f1_res, spectral_f1_res = getTraditionalClustering(data_xy, residuals, k)


    def getTraditionalClustering(self, data_xy, residuals, k):
        labels_true = np.array([0] * int(self.attributes[2]) + [1] * int(self.attributes[3]))
        ### Compute GMM Accuracy on (X, y) ###
        gmm_xy = GaussianMixture(n_components=k, random_state=19)
        gmm_xy.fit(data_xy)
        labels_xy = gmm_xy.predict(data_xy)

        gmm_acc_case1 = np.mean((labels_xy == 1) == labels_true)
        gmm_acc_case2 = np.mean((labels_xy == 0) == labels_true)
        gmm_acc = max(gmm_acc_case1, gmm_acc_case2)

        gmm_f1 = self.compute_f1(labels_xy, labels_true)

        ### Compute GMM Accuracy on Residuals ###
        gmm_res = GaussianMixture(n_components=k, random_state=19)
        gmm_res.fit(residuals.reshape(-1, 1))
        labels_res = gmm_res.predict(residuals.reshape(-1, 1))

        gmm_acc_res_case1 = np.mean((labels_res == 1) == labels_true)
        gmm_acc_res_case2 = np.mean((labels_res == 0) == labels_true)
        gmm_acc_res = max(gmm_acc_res_case1, gmm_acc_res_case2)

        gmm_f1_res = self.compute_f1(labels_res, labels_true)

        ### Compute K-Means Accuracy on (X, y) ###
        kmeans_xy = KMeans(n_clusters=k, random_state=19)
        kmeans_xy.fit(data_xy)
        kmeans_labels_xy = kmeans_xy.labels_

        kmeans_acc_case1 = np.mean((kmeans_labels_xy == 1) == labels_true)
        kmeans_acc_case2 = np.mean((kmeans_labels_xy == 0) == labels_true)
        kmeans_acc = max(kmeans_acc_case1, kmeans_acc_case2)

        kmeans_f1 = self.compute_f1(kmeans_labels_xy, labels_true)

        ### Compute K-Means Accuracy on Residuals ###
        kmeans_res = KMeans(n_clusters=k, random_state=19)
        kmeans_res.fit(residuals.reshape(-1, 1))
        kmeans_labels_res = kmeans_res.labels_

        kmeans_acc_res_case1 = np.mean((kmeans_labels_res == 1) == labels_true)
        kmeans_acc_res_case2 = np.mean((kmeans_labels_res == 0) == labels_true)
        kmeans_acc_res = max(kmeans_acc_res_case1, kmeans_acc_res_case2)

        kmeans_f1_res = self.compute_f1(kmeans_labels_res, labels_true)

        ### Compute Spectral Clustering Accuracy on (X, y) ###
        spectral_xy = SpectralClustering(n_clusters=k, random_state=19)
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

    def compute_f1(self, predicted_labels, labels_true):
        """Compute the best F1-score considering both direct and switched alignments."""
        # Case 1: Direct alignment
        f1_case1 = f1_score(labels_true, predicted_labels)

        # Case 2: Switched alignment
        f1_case2 = f1_score(labels_true, 1 - predicted_labels)  # Flip labels

        return max(f1_case1, f1_case2)

