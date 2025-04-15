from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.interventioninheritance import generate_data

def run_code(n, orig, intv, intv_type, f, clusters, k, r):
    # test folder name
    foldername = f"test_{n}_{orig}_{intv}_{intv_type}_{f}_{clusters}_{k}_{r}_nointv"

    # Data gen filepath
    dg_filepath = f"./inheritancetests/{foldername}"

    # Filepath
    filepath = f"./inheritancetests/{foldername}"

    # Generate the test data
    generate_data(dg_filepath, n, orig, intv, intv_type, f, clusters)

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(filepath, n, k, [], r, False )

def main():

    n = 30        # Number of tests
    orig = 2000       # Number of original datapoints
    intv = 1000      # Number of intervention datapoints
    intv_type = 0  # Type of intervention (0 to 4)
    f = 1         # Function type (0 to 2)
    clusters = 1   # Number of clusters for data generation
    k = 2         # Number of clusters for method
    r = 0          # Random flag
    run_code(n, orig, intv, intv_type, f, clusters, k, r)

    #ARI (Adjusted Rand Index)
    #NMI (Normalized Mutual Information)
    #FMI (Fowlkes-Mallows Index)

    # Generate stats using the provided arguments
    #spot.generate_stats(filepath, n, orig, intv, intv_type)  # wth ?
 #  15:03 srun with partition

if __name__ == "__main__":
    main()