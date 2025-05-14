'''
This file is unfinnished. POTENTIAL OF REMOVAL
'''
import numpy as np;
from combinator import Combinator;
from edge import Edge;
#from sampler import Sampler;
from dataTransformer import DataTransformer
from cc import CC
from node import Node;
import time
import RFunctions as rf
import multiprocessing



class CCWrapper:
    def __init__(self, dims=0, M=2):
        #self.sampler = Sampler();
        self.Transformer = DataTransformer(True);
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.M = M;
        self.spot = None
        print("Max interactions set to degree ", self.M);

    def generate_stats(self, file_path, n, k, needed_nodes, rand, mdl_th):
        filename = "/experiment"
        # Create a new stats file
        stats_file_name = f"STATS{n}.txt"
        stats_file = f"{file_path}/{stats_file_name}"
        with open(stats_file, "w") as stats:
            stats.write("id,intervention,dag_size,orig_data,intv_data,TP,TN,FP,FN,intv_acc,avg_ari_score,elapsed_time,method_time,gmm_time,kmeans_time,spectral_time,gmm_res_time,kmeans_res_time,spectral_res_time\n")

        total_intv_acc = 0
        total_final_ari_scores = []
        # Create a Pool of worker processes
        print(f"!!!!!!!!!!!CPUS = {multiprocessing.cpu_count()}!!!!!!!!!!!")
        pool = multiprocessing.Pool(processes=16)  # Use all available CPUs

        # Prepare the arguments for each test case
        tasks = [(i, file_path, n, k, needed_nodes, rand, mdl_th, stats_file) for i in range(n)]

        # Map the tasks to the processes in the pool
        results = pool.starmap(self.process_test_case, tasks)

        # After all the tasks are done, close the pool
        pool.close()
        pool.join()
        # Aggregate results
        for intv_found_acc, final_ari_scores, TP, TN, FP, FN in results:
            total_intv_acc += intv_found_acc
            total_final_ari_scores.extend(final_ari_scores)

        print(f"Stats file written to: {stats_file}")
        print(f"Total intervention accuracy {total_intv_acc} and Total final NMI scores {total_final_ari_scores}")

    def process_test_case(self, i, file_path, n, k, needed_nodes, rand, mdl_th, stats_file):
        filename1 = file_path + "/experiment" + str(i + 1)
        print(f"FILE NAME {filename1}")
        attributes_file = f"{filename1}/attributes1.txt"

        # Read attributes file
        with open(attributes_file, "r") as atts:
            lines = atts.readlines()
            values = lines[1].strip()  # Second line contains the values
            attributes = values.split(", ")

        # Start time for the test case
        start_time = time.time()

        # Get interventions found and accuracies
        intv_found_acc, final_ari_scores, TP, TN, FP, FN, runtimes = self.idk(filename1, int(attributes[1]), k, needed_nodes,
                                                                    rand, mdl_th)

        # End time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        # Unpack runtime
        (
            method_runtime, gmm_runtime, kmeans_runtime, spectral_runtime,
            gmm_res_runtime, kmeans_res_runtime, spectral_res_runtime
        ) = (
            runtimes.get(key, []) for key in [
            'cc', 'gmm', 'kmeans', 'spectral', 'gmm_res', 'kmeans_res', 'spectral_res'
        ]
        )
        # {'cc': 0, 'gmm': 0, 'kmeans': 0, 'spectral': 0, 'gmm_res': 0, 'kmeans_res': 0, 'spectral_res': 0}
        # Write results to the stats file
        with open(stats_file, "a") as stats:
            avg_ari_score = (sum(final_ari_scores) / len(final_ari_scores)) if final_ari_scores else 0
            stats.write(
                f"{i + 1},{attributes[0]},{attributes[1]},{attributes[2]},{attributes[3]},{TP:.2f},{TN:.2f},{FP:.2f},{FN:.2f},{intv_found_acc:.2f},{avg_ari_score:.2f},{elapsed_time:.4f},{method_runtime:.4f},{gmm_runtime:.4f},{kmeans_runtime:.4f},{spectral_runtime:.4f},{gmm_res_runtime:.4f},{kmeans_res_runtime:.4f},{spectral_res_runtime:.4f}\n"
            )

        return intv_found_acc, final_ari_scores, TP, TN, FP, FN

    def idk(self, filename, nodes, k, needed_nodes, rand, mdl_th):
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen
        self.cc = CC(Max_Interactions, log_results, verbose);
        self.cc.loadData(filename);
        found_intv, ari_scores, runtimes = self.cc.run(k, needed_nodes, rand, mdl_th);
        # runtimes {'cc': 0, 'gmm': 0, 'kmeans': 0, 'spectral': 0, 'gmm_res': 0, 'kmeans_res': 0, 'spectral_res': 0}
        try:
            intv_file = f"{filename}/interventions1.txt"
            intvs = np.loadtxt(intv_file, delimiter=',', dtype=int)
            if intvs.ndim == 0:  # Handle cases where intvs is scalar
                intvs = np.array([intvs])

        except Exception as e:
            print(f"An error occurred: {e}")

            # Ensure intvs is not empty
            if intvs.size == 0:
                print("No interventions found in the file.")
                return 0, []

        intv_cnt = intvs.shape[0]
        final_ari_scores = []

        true_labels = np.zeros(nodes)
        true_labels[intvs] = 1
        found_labels = np.zeros(nodes)
        found_labels[found_intv] = 1

        # Calculate TP, FP, FN, TN
        TP = np.sum((true_labels == 1) & (found_labels == 1))
        FP = np.sum((true_labels == 0) & (found_labels == 1))
        FN = np.sum((true_labels == 1) & (found_labels == 0))
        TN = np.sum((true_labels == 0) & (found_labels == 0))

        # Calculate Accuracy
        intv_accuracy = (TP + TN) / nodes

        for i in range(len(found_intv)):
            intv = found_intv[i]
            if intv in intvs:
                final_ari_scores.append(ari_scores[i])


        print(str(TP) + " INTERVENTIONS FOUND OUT OF " + str(intv_cnt) + " INTERVENTIONS")
        if TP == intv_cnt:
            print("ALL FOUND !!!")
        return intv_accuracy, final_ari_scores, TP, TN, FP, FN, runtimes

        #spot.analyzeLabels()

