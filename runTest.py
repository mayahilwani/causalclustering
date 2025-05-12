from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data

def run_code(n, orig, intv, intv_type, f, clusters, k, r, mdl_th, intv_strength, noise_level):
    # test folder name
    foldername = f"test_{n}_{orig}_{intv}_{intv_type}_{f}_{clusters}_{k}_{r}_{mdl_th}_{intv_strength}_{noise_level}"

    # Data gen filepath
    dg_filepath = f"/Users/mayahilwani/PycharmProjects/tests/{foldername}"

    # Filepath
    filepath = f"/Users/mayahilwani/PycharmProjects/tests/{foldername}" #f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Generate the test data
    generate_data(dg_filepath, n, orig, intv, intv_type, f, clusters, intv_strength, noise_level)

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(filepath, n, k, [], r, mdl_th )

def main():
   # 2500   2000-500  2250-250
    n = 1        # Number of tests
    orig = 2250       # Number of original datapoints
    intv = 250      # Number of intervention datapoints
    intv_type = 1   # Type of intervention (0 to 3)
    f = 2 # Function type (0 to 2) 0: linear,  1:poly degree2 , 2: periodic
    clusters = 2   # Number of clusters for data generation
    k = 2         # Number of clusters for method
    r = 0          # Random flag
    mdl_th = 0
    intv_strength = 2  # 0, 1 or 2
    noise_level = 1 # 0, 1 or 2
    run_code(n, orig, intv, intv_type, f, clusters, k, r, mdl_th, intv_strength, noise_level)
    # sbatch 50 2000 500 0 0 2 3 0 0 0 0
    # script.py 50 2000 500 0 1 2 4 0 0
    #ARI (Adjusted Rand Index)
    #NMI (Normalized Mutual Information)
    #FMI (Fowlkes-Mallows Index)
    # My File not changed

    # Generate stats using the provided arguments
    #spot.generate_stats(filepath, n, orig, intv, intv_type)  # wth ?
 #  15:03 srun with partition

if __name__ == "__main__":
    main()