from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data
import numpy as np
import matplotlib.pyplot as plt

def run_code(n, k, r, mdl_th):
    # Filepath
    datapath = f"/Users/mayahilwani/PycharmProjects/msc-mhilwani/real_tests" #f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(datapath, n, k, [1], r, mdl_th )

def main():

    #run_code(1,2,False, False)
    data_file = "full_data.txt"
    data = np.loadtxt(data_file, delimiter=',')

    '''# (3, 8, 5)
    # Select intervention samples (experiment 5)
    intv_data_3_8 = data[data[:, 11] == 5]

    # Select the rest (non-intervention on experiment 5)
    non_intv_data_3_8 = data[data[:, 11] != 5]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot non-intervention points in light gray
    ax.scatter(non_intv_data_3_8[:, 8], non_intv_data_3_8[:, 3], alpha=0.2, label="Observational data", color="gray")

    # Plot intervention points in a distinct color
    ax.scatter(intv_data_3_8[:, 8], intv_data_3_8[:, 3], alpha=0.6, label="Interventional data", color="blue")

    # Labels and title
    ax.set_title('Variable 3 vs Parent Variable 8 (highlighting intervention)')
    ax.set_xlabel('Parent Variable 8')
    ax.set_ylabel('Variable 3')
    ax.legend()
    plt.tight_layout()
    plt.show()'''

    # (1, 5, 6)
    # Select intervention samples (experiment 5)
    intv_data_1_5 = data[data[:, 11] == 6]

    # Select the rest (non-intervention on experiment 5)
    non_intv_data_1_5 = data[data[:, 11] != 6]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot non-intervention points in light gray
    ax.scatter(non_intv_data_1_5[:, 5], non_intv_data_1_5[:, 1], alpha=0.2, label="Other data", color="gray")

    # Plot intervention points in a distinct color
    ax.scatter(intv_data_1_5[:, 5], intv_data_1_5[:, 1], alpha=0.6, label="Interventional data", color="blue")

    # Labels and title
    ax.set_title('Variable 1 vs Parent Variable 5 (highlighting intervention)')
    ax.set_xlabel('Parent Variable 5')
    ax.set_ylabel('Variable 1')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()