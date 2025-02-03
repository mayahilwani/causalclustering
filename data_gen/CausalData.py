import cdt.data as tb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dgen import *

np.random.seed(12)

def main():
	base_path="./nn/"
	mechanisms=['nn']
	noises = ['uniform']
	node=[5,10,15]

	offset=0
	batch=25
	md=0.2

	for nd in node:
		for mec in mechanisms:
			for n in noises:
				print("Node: ",nd,": ",mec,", ",n, "Density: ",md*nd)
				for i in range(batch):
					generator = tb.AcyclicGraphGenerator(mec,nodes=nd, npoints=10000, noise=n, noise_coeff=0.3,dag_type='erdos',expected_degree=md*nd)
					data , G = generator.generate()
					graph=nx.to_numpy_array(G).astype(int)
					#data=gen_data(graph,10000)
					print(data.shape," : ",np.sum(graph))
					print(base_path+'data'+str(offset+i+1) +".txt")
					np.savetxt(base_path+'data'+str(offset+i+1) +".txt"	  , data , delimiter=',',fmt='%0.4f')   # X is an array
					np.savetxt(base_path+'data'+str(offset+i+1) +"_truth.txt", graph, delimiter=',',fmt='%d')
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
