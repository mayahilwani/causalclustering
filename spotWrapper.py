'''
This file is unfinnished. POTENTIAL OF REMOVAL
'''
import numpy as np;
from combinator import Combinator;
from edge import Edge;
#from sampler import Sampler;
from dataTransformer import DataTransformer
from spot import Spot;
from node import Node;
import time
import RFunctions as rf



class SpotWrapper:
    def __init__(self, slp, dims=0, M=2):
        self.slope_ = slp;
        #self.sampler = Sampler();
        self.Transformer = DataTransformer(True);
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.M = M;
        self.spot = None
        print("Max interactions set to degree ", self.M);

    def generate_stats(self, file_path, n):
        filename = "/expirement"
        # Create a new stats file
        stats_file = f"{file_path}/STATS5_spec.txt"
        with open(stats_file, "w") as stats:
            stats.write("id, intervention, dag_size, orig_data, intv_data, TP, TN, FP, FN, intv_acc, avg_cluster_acc, elapsed_time\n")

        total_intv_acc = 0
        total_final_accuracies = []

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
            intv_found_acc, final_accuracies, TP, TN, FP, FN = self.idk(filename1, int(attributes[1]))
            # END TIME
            end_time = time.time()  # Record the end time
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Write to the stats file for each test case
            with open(stats_file, "a") as stats:
                avg_accuracy = (
                    sum(final_accuracies) / len(final_accuracies) if final_accuracies else 0
                )#ADD TIME TO THE FILE AT THE END OF THE LINE
                stats.write(
                    f"{i + 1},{attributes[0]}, {attributes[1]}, {attributes[2]}, {attributes[3]}, {TP:.2f}, {TN:.2f}, {FP:.2f}, {FN:.2f}, {intv_found_acc:.2f}, {avg_accuracy:.2f}, {elapsed_time:.4f}\n"
                )

            # Update totals for averages
            total_intv_acc += intv_found_acc
            total_final_accuracies.extend(final_accuracies)

        # Calculate and write overall averages to the stats file
        overall_intv_acc = total_intv_acc / n
        overall_avg_accuracy = (
            sum(total_final_accuracies) / len(total_final_accuracies)
            if total_final_accuracies
            else 0
        )
        #with open(stats_file, "a") as stats:
         #   stats.write("\n")
          #  stats.write(f"Overall Interventions Found Accuracy: {overall_intv_acc:.2f}\n")
           # stats.write(f"Overall Average Accuracy: {overall_avg_accuracy:.2f}\n")

        print(f"Stats file written to: {stats_file}")

    def idk(self, filename, nodes):
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen
        self.spot = Spot(Max_Interactions, log_results, verbose);
        self.spot.loadData(filename);
        found_intv, accuracies = self.spot.run();
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
        final_accuracies = []

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
                final_accuracies.append(accuracies[i])


        print(str(TP) + " INTERVENTIONS FOUND OUT OF " + str(intv_cnt) + " INTERVENTIONS")
        if TP == intv_cnt:
            print("ALL FOUND !!!")
        #intv_acc = intv_found/intv_cnt
        return intv_accuracy, final_accuracies, TP, TN, FP, FN

        #spot.analyzeLabels()

