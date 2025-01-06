from sklearn.mixture import GaussianMixture
from slope import Slope;
from utils import *
import numpy as np;
from top_sort import *
import RFunctions as rf
import matplotlib
matplotlib.use('TkAgg')


def gmm_prob(data, gt, slope_):
    normalized_vars = Standardize(data)
    recs = normalized_vars.shape[0]
    dim = normalized_vars.shape[1]
    headers = [i for i in range(0, dim)]
    dims = gt.shape[1]
    g = Graph(dims)
    # Construct the Edges from the ground truth (gt_network)
    for i in range(dims):
        for j in range(dims):
            if gt[i, j] == 1:
                g.addEdge(i, j)
                #Edges[i][j] = Edge(i, j, [], 0)
    ordering = g.nonRecursiveTopologicalSort()
    #self.ordering = ordering
    print('ORDERING ' + str(ordering))

    # Traverse the nodes top to bottom
    for i, variable_index in enumerate(ordering):
        # Get parents of the node
        #print('Iteration: ' + str(i) + ' and Node is: ' + str(variable_index))
        pa_i = np.where(gt[:, variable_index] == 1)[0]
        print(f"NODE {variable_index} has parents {pa_i}")

        # If node has no parents print and skip it
        if len(pa_i) == 0:
            print(f"Variable {variable_index} has no parents.")
            continue

        X = data[:, pa_i]
        y = data[:, variable_index]

        # Fit MARS on the entire dataset
        sse, score, coeff, hinge_count, interactions, rearth = slope_.FitSpline(X, y)
        y_pred = rf.predict_mars(X, rearth)
        residuals = y - y_pred
        residuals = residuals.reshape(-1, 1)
        # Fit GMM for k=1 and k=2
        gmm_k1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm_k1.fit(residuals)
        gmm_k2 = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm_k2.fit(residuals)

        # Compute probabilities for k=1
        log_prob_k1 = gmm_k1.score_samples(residuals)  # Log(P(d))
        total_log_likelihood_k1 = np.sum(log_prob_k1)  # Sum of log(P(d)) over all data points

        # Compute probabilities for k=2
        log_prob_k2 = gmm_k2.score_samples(residuals)  # Log(P(d))
        total_log_likelihood_k2 = np.sum(log_prob_k2)  # Sum of log(P(d)) over all data points

        # Output the results
        print(f"Total Log-Likelihood for k=1: {total_log_likelihood_k1}")
        print(f"Total Log-Likelihood for k=2: {total_log_likelihood_k2}")

        # Compare the models
        if total_log_likelihood_k1 > total_log_likelihood_k2:
            print('k=1 fits node: ' + str(variable_index) +  ' better.')
        else:
            print('k=2 fits node: ' + str(variable_index) +  ' better.')

        aic_k1 = gmm_k1.aic(residuals)
        aic_k2 = gmm_k2.aic(residuals)

        if aic_k1 < aic_k2:
            print('k=1 fits node better (based on AIC).')
        else:
            print('k=2 fits node better (based on AIC).')



def main():
    try:
        base_path = "C:/Users/ziadh/Documents/CausalGen-Osman/maya_data/diffSizes/experiment4010flip"
        gt_file = f"{base_path}/truth1.txt"
        gt = np.loadtxt(gt_file, delimiter=',')
        data_file1 = f"{base_path}/data1.txt"
        data_file2 = f"{base_path}/dataintv1.txt"
        data_file3 = f"{base_path}/interventions1.txt"
        data1 = np.loadtxt(data_file1, delimiter=',')
        data2 = np.loadtxt(data_file2, delimiter=',')
        interventions = np.loadtxt(data_file3, delimiter=',')
        if data1.shape[1] != data2.shape[1]:
            raise ValueError("The two files must have the same number of columns for vertical concatenation.")
        data = np.vstack((data1, data2))
        print(data.shape)
        rn = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        n = 2500
        slope_ = Slope()
        gmm_prob(data, gt, slope_)
        print(interventions)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()