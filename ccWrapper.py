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
            stats.write("id, intervention, dag_size, orig_data, intv_data, TP, TN, FP, FN, intv_acc, avg_ari_score, elapsed_time\n")

        total_intv_acc = 0
        total_final_ari_scores = []

        # Loop through the test cases
        for i in range(n):
            filename1 = file_path + filename + str(i + 1)
            print("FILE NAME " + str(filename1))
            attributes_file = f"{filename1}/attributes1.txt"
            with open(attributes_file, "r") as atts:
                lines = atts.readlines()
                values = lines[1].strip()  # Second line contains the values
                # Convert the values to a list (optional)
                attributes = values.split(", ")
            # START TIME
            start_time = time.time()  # Record the start time
            # Get interventions found and accuracies
            intv_found_acc, final_ari_scores, TP, TN, FP, FN = self.idk(filename1, int(attributes[1]), k, needed_nodes, rand, mdl_th)
            # END TIME
            end_time = time.time()  # Record the end time
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Write to the stats file for each test case
            with open(stats_file, "a") as stats:
                avg_ari_score = (
                    sum(final_ari_scores) / len(final_ari_scores) if final_ari_scores else 0
                )#ADD TIME TO THE FILE AT THE END OF THE LINE
                stats.write(
                    f"{i + 1},{attributes[0]}, {attributes[1]}, {attributes[2]}, {attributes[3]}, {TP:.2f}, {TN:.2f}, {FP:.2f}, {FN:.2f}, {intv_found_acc:.2f}, {avg_ari_score:.2f}, {elapsed_time:.4f}\n"
                )

            # Update totals for averages
            total_intv_acc += intv_found_acc
            total_final_ari_scores.extend(final_ari_scores)

        print(f"Stats file written to: {stats_file}")
        print(f"Total intervention accuracy {total_intv_acc}  and Total final NMI scores {total_final_ari_scores}")

    def idk(self, filename, nodes, k, needed_nodes, rand, mdl_th):
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen
        self.cc = CC(Max_Interactions, log_results, verbose);
        self.cc.loadData(filename);
        found_intv, ari_scores = self.cc.run(k, needed_nodes, rand, mdl_th);
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
        return intv_accuracy, final_ari_scores, TP, TN, FP, FN

        #spot.analyzeLabels()

