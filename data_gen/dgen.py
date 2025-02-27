from top_sort import *
import numpy as np
from dataTransformer import DataTransformer


def gen_data(gt,num_samples=1000,intv_list=[],pre_config=None):
	dims = gt.shape[1]
	data = np.zeros((num_samples,dims))
	transf = DataTransformer(True);

	g= Graph(dims)
	for i in range(dims):
		for j in range(dims):
			if gt[i,j]==1:
				g.addEdge(i, j);
	
	ordering = g.nonRecursiveTopologicalSort()
	# ADD FUNC HERE IF ADDED IN dataTransformer.py
	f = [ transf.poly  ] # , transf.osc
	intv_type = [transf.intv_f, transf.intv_s, transf.intv_r, transf.intv_sh] # , transf.intv_sh
	intv_name = ['flip', 'scale', 'random', 'shift']
	init_set=[False for i in range(dims)]
	f_id = np.random.randint(0,len(f))
	intv_id = 0 #np.random.randint(0,len(intv_type))
	fx = f[f_id]
	intv = "None"
	configs={}
	for variable in ordering:
		fx=f[f_id]		
		if variable in intv_list: #####
			fx = intv_type[intv_id]
			intv = intv_name[intv_id]

		pa_i = np.where(gt[:,variable]==1)[0]

		x = np.ones(num_samples).reshape(-1,1)
		parents_exist=False
		if len(pa_i)!=0:
			x = data[:,pa_i]
			parents_exist=True
		curr_config = None if pre_config is None else pre_config[variable]
		tot,cfg = fx(x,num_samples,parents_exist,curr_config)
		data[:,variable]=tot.reshape(-1)
		configs[variable]=cfg
		init_set[variable]=True
	#print('CONFIGS : ' + str(configs))
	return data,configs,intv
	
'''
def main():
	gt = np.eye(6) * 0
	gt[0,2]=1
	gt[1,3]=1
	gt[2,3]=1
	gt[1,5]=1
	gt[4,5]=1
	dt = gen_data(gt,10000)
	#import ipdb;ipdb.set_trace()

main()
#'''
