from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data

def run_code(n, k, r, mdl_th):
    # Filepath
    datapath = f"/Users/mayahilwani/PycharmProjects/msc-mhilwani/colon_data" #f"C:/Users/ziadh/Documents/..MAYAMSC/results/tests/{foldername}"

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(datapath, n, 2, [1], r, mdl_th )

def main():

    run_code(12,2,False, False)


if __name__ == "__main__":
    main()