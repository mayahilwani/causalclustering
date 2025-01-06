'''
THIS FILE IS FOR TESTING THE SCORING
'''
from spotWrapper import SpotWrapper;
import numpy as np
import matplotlib.pyplot as plt
import RFunctions as rf
from slope import Slope;
from node import Node;
from utils import *
import spot

def plot_scoring(rn, gt, data):
    print('VARS BEFORE:  ' + str(data.shape))
    normalized_vars = Standardize(data)
    print('VARS AFTER:  ' + str(normalized_vars.shape))
    recs = normalized_vars.shape[0]
    dim = normalized_vars.shape[1]
    headers = [i for i in range(0, dim)]
    slope = Slope()

    # Initialize Slope and Spot objects
    spot_ = spot.Spot(2, dims=dim)
    #globe_ = globe.Globe(slope, dims=dim, M=2)
    for variable_index in rn:
        print('NODE: ' + str(variable_index))
        pa_i = np.where(gt[:, variable_index] == 1)[0]
        print('PARENTS: ' + str(pa_i))
        X = data[:, pa_i].reshape(-1, len(pa_i))
        #print( 'X shape  ' + str(X.shape))
        #X = data[:, pa_i]
        y = data[:, variable_index]
        #print( 'y shape  ' + str(y.shape))
        sse, score, coeff, hinge_count, interactions, rearth = slope.FitSpline(X,y)
        #print(' Global spline calculated! ')
        curr_node = Node(normalized_vars[:, variable_index].reshape(recs, -1), spot_)
        #print(' node done.')
        # Define indices for the two sections of the data
        i_g1 = np.arange(5000)  # First 5000 data points
        interv_points = len(data) - 5000
        i_g2 = np.arange(len(data) - interv_points, len(data))  # Last X data points

        X_g1 = data[np.ix_(i_g1, pa_i)]
        y_g1 = data[i_g1, variable_index]

        X_g2 = data[np.ix_(i_g2, pa_i)]
        y_g2 = data[i_g2, variable_index]
        #print("Shape of X_g1:", X_g1.shape)
        #print("Shape of y_g1:", y_g1.shape)
        #print("Shape of X_g2:", X_g2.shape)
        #print("Shape of y_g2:", y_g2.shape)

        sse1, score1, coeff1, hinge_count1, interactions1, rearth1 = slope.FitSpline(X_g1,y_g1)
        sse2, score2, coeff2, hinge_count2, interactions2, rearth2 = slope.FitSpline(X_g2,y_g2)
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen

        #spotw = SpotWrapper(Max_Interactions, log_results, verbose, dims=dim);
        score_all = spot_.ComputeScore(hinge_count, interactions, sse, score, len(y), curr_node.min_diff,
                                      np.array([len(pa_i)]), show_graph=False)
        print('SCORE for initial model is ' + str(score_all))

        rows1 = 5000
        rows2 = interv_points
        print(str(rows1) + ' members in group 1  and ' + str(rows2) + ' members in group 2')
        score_split = spot_.ComputeScoreSplit(hinge_count1, interactions1, sse1, score1, rows1, hinge_count2,
                                             interactions2, sse2, score2, rows2, curr_node.min_diff,
                                             np.array([len(pa_i)]), show_graph=False)
        print('SCORE for splitting model is ' + str(score_split))

        if score_all > score_split:
            print(f"Splitting model is better with score  {score_all - score_split}")
        else:
            print(f"Original model is better with score {score_split - score_all}")

def main():
    try:
        base_path = "C:/Users/ziadh/Documents/CausalGen-Osman/maya_data/shift/experiment5"
        gt_file = f"{base_path}/truth1.txt"
        gt = np.loadtxt(gt_file, delimiter=',')
        data_file1 = f"{base_path}/data1.txt"
        data_file2 = f"{base_path}/dataintv1.txt"
        data1 = np.loadtxt(data_file1, delimiter=',')
        data2 = np.loadtxt(data_file2, delimiter=',')
        if data1.shape[1] != data2.shape[1]:
            raise ValueError("The two files must have the same number of columns for vertical concatenation.")
        data = np.vstack((data1, data2))
        print(data.shape)
        rn = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        plot_scoring(rn, gt, data)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()