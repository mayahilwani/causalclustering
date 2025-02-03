import cdt.data as tb
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dgen import *

np.random.seed(90001)

def main():
	base_path="./mech_data/experiment"
	mechanisms=['nn']
	noises = ['uniform']
	node=[5,10,15]

	offset=0
	batch=20
	md=0.2

	n_mech=3
	n_samples=5000

	for nd in node:
		for mec in mechanisms:
			for n in noises:
				print("Node: ",nd,": ",mec,", noise: ",n, "Density: ",md*nd)
				for i in range(batch):
					store_path = base_path + str(offset+i+1) + "/"
					if not os.path.exists(store_path):
							os.makedirs(store_path);

					for d_counter in range(1,n_mech+1):
						generator = tb.AcyclicGraphGenerator(mec,nodes=nd, npoints=1, noise=n, noise_coeff=0.3,dag_type='erdos',expected_degree=md*nd)
						_ , G = generator.generate()
						graph=nx.to_numpy_array(G).astype(int)
						#generate different mechanisms
						data,_=gen_data(graph,n_samples)
						print(data.shape," : ",np.sum(graph))
						print(store_path+"data"+ str(d_counter) +".txt")
						np.savetxt(store_path+"data" + str(d_counter) + ".txt"	, data  , delimiter=',' , fmt='%0.7f')   # X is an array
						np.savetxt(store_path+"truth"+ str(d_counter) + ".txt"	, graph , delimiter=',' , fmt='%d')



				offset+=batch
		


'''
def test_truth(data):
	counter=0
	while counter<100:
		counter+=1
		x= data[:,int(input("x:"))]
		y= data[:,int(input("y:"))]
		plt.scatter(x, y)	
		plt.show()
		#import ipdb;ipdb.set_trace()
#'''
main();
