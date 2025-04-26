from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data

def run_code(n, orig, intv, intv_type, f, clusters, k, r, mdl_th):
    # test folder name
    foldername = f"test_{n}_{orig}_{intv}_{intv_type}_{f}_{clusters}_{k}_{r}_NOoutliers"

    # Data gen filepath
    dg_filepath = f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Filepath
    filepath = f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Generate the test data
    generate_data(dg_filepath, n, orig, intv, intv_type, f, clusters)

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(filepath, n, k, [], r, mdl_th )

def main():

    n = 1        # Number of tests
    orig = 2000       # Number of original datapoints
    intv = 500      # Number of intervention datapoints
    intv_type = 0  # Type of intervention (0 to 4)
    f = 1         # Function type (0 to 2)
    clusters = 2   # Number of clusters for data generation
    k = 2         # Number of clusters for method
    r = 0          # Random flag
    mdl_th = 0
    run_code(n, orig, intv, intv_type, f, clusters, k, r, mdl_th)
    # script.py 50 2000 500 0 1 2 4 0 0
    #ARI (Adjusted Rand Index)
    #NMI (Normalized Mutual Information)
    #FMI (Fowlkes-Mallows Index)

    # Generate stats using the provided arguments
    #spot.generate_stats(filepath, n, orig, intv, intv_type)  # wth ?
 #  15:03 srun with partition

if __name__ == "__main__":
    main()