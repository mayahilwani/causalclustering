from copy import deepcopy
import cdt.data as tb
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from data_gen.dgen import *
from data_gen.dataTransformer import DataTransformer

#np.random.seed(90001)
np.random.seed(80300)

def generate_data(filepath, n, orig, intv, intv_type, f, c=2):
    """
    Generate data based on the provided parameters.

    Args:
        n (int): Number of tests.
        orig (int): Number of original datapoints.
        intv (int): Number of intervention datapoints.
        intv_type (int): Type of intervention (0 to 4).
        f (int): Function type (0 to 2).
        c (int): Number of clusters.
    """
    # Filepath (you can modify this as needed)
    base_path = f"{filepath}/experiment"
    mechanism = ['polynomial']
    noise = ['uniform']
    nodes = [3]  # num nodes
    offset = 0  # ?
    batch = n  # Use the number of tests (n) for the batch size
    md = 0.2  # ???
    dt = DataTransformer(True)
    n_intv = c-1
    n_obs = 1  # ?
    n_samples = orig  # Use the number of original datapoints (orig)

    for nd in nodes:
        for mec in mechanism:
            for n in noise:
                print("Node: ", nd, ": ", mec, ", noise: ", n, "Density: ", md * nd)
                for i in range(batch):
                    d_counter = 1
                    store_path = base_path + str(offset + i + 1) + "/"
                    if not os.path.exists(store_path):
                        os.makedirs(store_path)
                    # Manually define the DAG with 3 nodes: 0 -> 1 -> 2
                    graph = np.array([
                        [0, 1, 0],  # Node 0 → Node 1
                        [0, 0, 1],  # Node 1 → Node 2
                        [0, 0, 0]  # Node 2 → No outgoing edges
                    ])
                    # Generate obs data
                    #gt,num_samples=1000,intv_list=[],intv_id = 0, f_id = 0, pre_config = None
                    data1, pre_config, _ = gen_data(graph, n_samples, [], intv_type, f)
                    np.savetxt(store_path + "origdata" + str(d_counter) + ".txt", data1, delimiter=',', fmt='%0.7f')
                    np.savetxt(store_path + "truth" + str(d_counter) + ".txt", graph, delimiter=',', fmt='%d')

                    # Generate INTERVENTION data
                    all_data = [data1]  # Start with original data
                    valid_intv_list = [1]  # To ensure consistency across interventions

                    for i in range(n_intv):  # Loop for c-1 interventions
                        new_n_samples = intv  # Number of intervention datapoints

                        data2, _, intv_name = gen_data(graph, new_n_samples, valid_intv_list, intv_type, f, pre_config)

                        if all_data[0].shape[1] != data2.shape[1]:
                            raise ValueError(
                                "The two files must have the same number of columns for vertical concatenation.")

                        all_data.append(data2)  # Append new intervention data
                        print(f"Intervention {i + 1}: {valid_intv_list}")

                    # Stack all data together and normalize
                    all_data = np.vstack(all_data)
                    all_data_normalized = dt.normalize_data(all_data)

                    print('SHAPE IS : ', str(all_data_normalized.shape))
                    print(store_path + "data" + str(d_counter) + ".txt")
                    np.savetxt(store_path + "data" + str(d_counter) + ".txt", all_data_normalized, delimiter=',',
                               fmt='%.7f')

                    if len(valid_intv_list) == 0:
                        np.savetxt(store_path + "interventions" + str(d_counter) + ".txt", [], delimiter=',', fmt='%d')
                    else:
                        np.savetxt(store_path + "interventions" + str(d_counter) + ".txt", [valid_intv_list],
                                   delimiter=',', fmt='%d')

                    attributes_file = f"{store_path}attributes1.txt"
                    with open(attributes_file, "w") as atts:
                        atts.write("intervention_type, num_nodes, orig_data, intv_data, num_interventions\n")
                        atts.write(f"{intv_name}, {str(nd)}, {str(n_samples)}, {str(new_n_samples)}, {str(n_intv)}\n")

def main():
    # Example usage
    filepath = 'mypath'
    n = 5  # Number of tests
    orig = 1000  # Number of original datapoints
    intv = 500  # Number of intervention datapoints
    intv_type = 2  # Type of intervention (0 to 4)
    f = 1  # Function type (0 to 2)

    # Call the data generation function
    generate_data(filepath, n, orig, intv, intv_type, f)

if __name__ == "__main__":
    main()