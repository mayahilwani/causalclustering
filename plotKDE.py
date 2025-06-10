from slope import Slope
import RFunctions as rf
import sys;
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def main():
    #filepath= "./tests/test_50_2000_500_1_2_2_4_0_0/"
    filepath = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/test_50_2000_500_3_0_2_3_0_0_1_0/" #/msc-mhilwani # test_1_2250_250_0_1_2_2_0_0_0_1

    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    slope = Slope()
    filename1 = filepath + str('experiment1')
    data_file = f"{filename1}/data1.txt"
    data = np.loadtxt(data_file, delimiter=',')
    node = 7
    parents = [0]
    # NO CLUSTERS  experiment1  7  [0]
    # CLUSTERS experiment12 5 [0]
    # fit a MARS from the parents to node and then get residuals and then plot the kde
    X = data[:, parents]
    y = data[:, node]

    sse, score, coeff, hinge_count, interaction, rearth = slope.FitSpline(X, y)
    y_pred = rf.predict_mars(X, rearth)
    residuals = y - y_pred
    '''# Plot KDE of residuals
    plt.figure(figsize=(8, 5))
    sns.kdeplot(residuals, fill=True)
    # Save as two-column CSV
    np.savetxt("residual_kde_data.csv", np.column_stack((x_vals, density(x_vals))), delimiter=",", header="x,density",
               comments='')
    plt.title("KDE of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()'''
    X_flat = X.flatten()
    # Sort by X for a smooth line
    sorted_indices = np.argsort(X_flat)
    X_sorted = X_flat[sorted_indices]
    y_sorted = y[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Save sorted data for plotting
    np.savetxt(
        "regression_fit_data_1.csv",
        np.column_stack((X_sorted, y_sorted, y_pred_sorted)),
        delimiter=",",
        header="x,y,y_pred",
        comments=''
    )
    '''
    # Compute KDE
    kde = gaussian_kde(residuals)

    # Choose x-axis range (usually slightly wider than min/max)
    x_vals = np.linspace(residuals.min() - 1, residuals.max() + 1, 1000)
    density_vals = kde(x_vals)

    # Save to CSV for LaTeX plotting
    np.savetxt(
        "residual_kde_data_1.csv",
        np.column_stack((x_vals, density_vals)),
        delimiter=",",
        header="x,density",
        comments=''
    )

    # Optional: preview the plot
    plt.plot(x_vals, density_vals)
    plt.title("KDE of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.show()'''

main();
