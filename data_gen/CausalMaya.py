from copy import deepcopy
import cdt.data as tb
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dgen import *
from dataTransformer import DataTransformer

#np.random.seed(90001)
np.random.seed(80300)

def main():
    base_path = "./norm/non_linear_2/test_2to5_2/expirement"
    mechanism = ['polynomial']
    noise = ['uniform']
    nodes = [10]  # num nodes
    offset = 0 # ?
    batch = 50 #?
    md = 0.2 #???
    md = 0.2 #???
    dt = DataTransformer(True)
    n_intv = 1
    n_obs = 1 #?
    n_samples = 2000
    for nd in nodes:
        for mec in mechanism:
            for n in noise:
                print("Node: ", nd, ": ", mec, ", noise: ", n, "Density: ", md * nd)
                for i in range(batch):
                    d_counter = 1
                    store_path = base_path + str(offset+i+1) + "/"
                    if not os.path.exists(store_path):
                        os.makedirs(store_path)
                    generator = tb.AcyclicGraphGenerator(mec, nodes=nd, npoints=1, noise=n, noise_coeff=0.3, dag_type='erdos', expected_degree=md * nd)
                    _, G = generator.generate()
                    graph = nx.to_numpy_array(G).astype(int)
                    # Enforce a maximum of 2 parents for each node
                    for node in range(graph.shape[1]):  # Iterate over all nodes
                        parents = np.where(graph[:, node] == 1)[0]  # Find parent nodes
                        if len(parents) > 2:
                            # Randomly select 2 parents to keep
                            parents_to_keep = np.random.choice(parents, size=2, replace=False)
                            # Remove other edges
                            for parent in parents:
                                if parent not in parents_to_keep:
                                    graph[parent, node] = 0
                    # Generate obs data
                    data1, pre_config, _ = gen_data(graph, n_samples)
                    #print(data1.shape, " : ", np.sum(graph)) # ????
                    #print(store_path+"data"+ str(d_counter) +".txt")
                    np.savetxt(store_path + "origdata" + str(d_counter) + ".txt", data1, delimiter=',', fmt='%0.7f')
                    np.savetxt(store_path + "truth" + str(d_counter) + ".txt", graph, delimiter=',', fmt='%d')
                    print('NOW INTERVENTION')
                    new_n_samples = 500
                    #for qq in range(n_intv):
                    #    print('qq is: '+ str(qq))
                        #d_counter += 1
                    #igraph = deepcopy(graph)
                    intv_count = np.random.randint(1, int(np.log2(nodes) + 1)) # 2
                    intv_list1 = list(np.random.choice(nd, intv_count, replace=False))
                    valid_intv_list = [intv for intv in intv_list1 if np.where(graph[:, intv] == 1)[0].size > 0] #gt[:,variable]==1)[0]
                    #intv_list = valid_intv_list

                    data2, _, intv = gen_data(graph, new_n_samples, valid_intv_list, pre_config)
                    if data1.shape[1] != data2.shape[1]:
                        raise ValueError("The two files must have the same number of columns for vertical concatenation.")
                    all_data = np.vstack((data1, data2))
                    all_data_normalized = dt.normalize_data(all_data) #all_data
                    print('SHAPE IS : ' , str(all_data_normalized.shape))
                    print(store_path+"data"+ str(d_counter) +".txt")
                    np.savetxt(store_path + "data" + str(d_counter) + ".txt", all_data_normalized, delimiter=',', fmt='%.7f')
                    print("Interventions: ", valid_intv_list)
                    if len(valid_intv_list) == 0:
                        np.savetxt(store_path + "interventions" + str(d_counter) + ".txt", [], delimiter=',', fmt='%d')
                    else:
                        np.savetxt(store_path + "interventions" + str(d_counter) + ".txt", [valid_intv_list], delimiter=',', fmt='%d')
                    attributes_file = f"{store_path}attributes1.txt"
                    with open(attributes_file, "w") as atts:
                            atts.write("intervention_type, num_nodes, orig_data, intv_data\n")
                            atts.write(f"{intv}, {str(nd)}, {str(n_samples)}, {str(new_n_samples)}\n")

                    # import ipdb;ipdb.set_trace()
                    #print(store_path + "dataintv" + str(d_counter) + ".txt")
                    #np.savetxt(store_path + "dataintv" + str(d_counter) + ".txt", data2, delimiter=',', fmt='%0.7f')  # X is an array
                    #np.savetxt(store_path + "truthintv" + str(d_counter) + ".txt", igraph, delimiter=',', fmt='%d')


if __name__ == "__main__":
    main()