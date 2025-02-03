import numpy as np;
import random

np.random.seed(42)

class DataTransformer:

	def __init__(self,inclusive):
		self.inc=inclusive;
		
	# HARD INTERVENTION 

	'''
	def intv(self,x,num_samples,parents_exist,pre_config):
		Y_val=np.random.normal(0,0.001,x.shape[0])
		return Y_val,pre_config
	'''
	
	# SOFT INTERVENTION
	def intv_f(self, x, num_samples, parents_exist, pre_config):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]
		new_config = pre_config
		print(f'OLD COEF: {coeff}')
		#new_coeff = coeff
        # Apply a transformation to the coefficients
		new_coeff = coeff.copy()
		new_coeff = new_coeff * np.random.uniform(-4, -0.5)
		Y_val = np.random.normal(0, 0.2, x.shape[0]).reshape(-1, 1)
		n = 0
		print(f'NEW COEF: {new_coeff}')
		print('f-ids:   ' + str(f_ids))
		tfs = []
		for i in range(dims):
			tfs.append(self.transform(x[:, i], f_ids[i]))
		dt = np.hstack(tfs)
		shift = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3)
		#print('PRINT in intv()')
		Y_val = np.dot(dt, new_coeff).reshape(-1, 1) + np.random.normal(0, 0.6, x.shape[0]).reshape(-1, 1)
		#mu_ 	= np.mean(y_vals);
		#sdev_ 	= np.std(y_vals);
		#Y_val 	= (y_vals - mu_) / sdev_;
		Y_val = Y_val + shift
		new_config = (new_coeff, f_ids)
		#print('Y_val shape from intv() : ' + str(Y_val.shape))
		return Y_val, new_config
	
	def intv_s(self, x, num_samples, parents_exist, pre_config):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]
		new_config = pre_config
		#new_coeff = coeff
        # Apply a transformation to the coefficients
		new_coeff = coeff.copy()
		new_coeff = new_coeff * np.random.uniform(1.5, 2.5)
		Y_val = np.random.normal(0, 0.2, x.shape[0]).reshape(-1, 1)
		print(f'OLD COEF: {coeff}')
		print(f'NEW COEF: {new_coeff}')
		print('f-ids:   ' + str(f_ids))
		tfs = []
		for i in range(dims):
			tfs.append(self.transform(x[:, i], f_ids[i]))
		dt = np.hstack(tfs)
		shift = np.random.choice([-1, 1]) * np.random.uniform(1.5, 2.5)
		print('PRINT in intv()')
		Y_val = np.dot(dt, new_coeff) + np.random.normal(0,2*x.shape[1],x.shape[0]) #np.random.normal(0, 0.2, x.shape[0])
		#mu_ 	= np.mean(y_vals);
		#sdev_ 	= np.std(y_vals);
		#Y_val 	= (y_vals - mu_) / sdev_;
		Y_val = Y_val + shift
		print('Y_val is changed. ')
		new_config = (new_coeff, f_ids)
		return Y_val, new_config
	
	def intv_r(self, x, num_samples, parents_exist, pre_config):
		coeff = np.array(pre_config[0])
		print(f'OLD COEF: {coeff}')
		transf = DataTransformer(True)
		fx = transf.poly
		tot,cfg = fx(x,num_samples,parents_exist,None)
		Y_val = tot.reshape(-1)
		new_config = cfg
		new_coeff = np.array(new_config[0])
		f_ids = new_config[1]
		print(f'NEW COEF: {new_coeff}')
		print('f-ids:   ' + str(f_ids))

		return Y_val, new_config

	def intv_sh(self, x, num_samples, parents_exist, pre_config):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]
		new_config = pre_config
		# new_coeff = coeff
		# Apply a transformation to the coefficients
		new_coeff = coeff.copy()
		#new_coeff = new_coeff * np.random.uniform(1.5, 5)
		go_on = True
		Y_val = np.random.normal(0, 0.2, x.shape[0]).reshape(-1, 1)

		print(f'OLD COEF: {coeff}')
		print(f'NEW COEF: {new_coeff}')
		print('f-ids:   ' + str(f_ids))
		if go_on:
			tfs = []
			for i in range(dims):
				tfs.append(self.transform(x[:, i], f_ids[i]))
			dt = np.hstack(tfs)
			shift = np.random.choice([-1, 1]) * np.random.uniform(10, 20)
			print('PRINT in intv()')
			Y_val = np.dot(dt, new_coeff) + np.random.normal(0, 2 * x.shape[1],
															 x.shape[0])  # np.random.normal(0, 0.2, x.shape[0])
			#mu_ 	= np.mean(Y_val);
			#sdev_ 	= np.std(Y_val);
			#Y_val 	= (Y_val - mu_) / sdev_;
			Y_val = Y_val + shift
			print('Y_val is changed. ')
			new_config = (new_coeff, f_ids)
		return Y_val, new_config

	def poly(self,x,num_samples,parents_exist,pre_config):
		Y_val=np.random.normal(0,3*x.shape[1],x.shape[0])
		dims = x.shape[1]
		#f_ids = np.random.randint(0,2,dims) # was 3
		#f_ids = np.zeros(dims, dtype=int)
		# Generate until at least one element is 1
		while True:
			f_ids = np.random.randint(0, 2, dims)
			if np.any(f_ids == 1):  # Check if there's at least one 1
				break
		# Set all to 0 !!!!!!!!!!!!!!!
		if pre_config is not None:
			f_ids = pre_config[1]

		tfs = []
		signed_coeffs=[]
		if parents_exist:
			for i in range(dims):
				tfs.append(self.transform(x[:,i],f_ids[i]))

			dt = np.hstack(tfs)
			new_dims = dt.shape[1]
			s_dict={};
			s_dict[0] = -1
			s_dict[1] =  1

			if pre_config is None:
				coeffs = np.random.uniform(2,5,new_dims)
				signs  = np.array( [ s_dict[ss] for ss in list(np.random.randint(0,2,new_dims))  ]  )
				signed_coeffs = coeffs * signs
			else:
				signed_coeffs = pre_config[0]

			Y_val 	= np.dot(dt,signed_coeffs)+np.random.normal(0,2*x.shape[1],x.shape[0])
			#mu_ 	= np.mean(Y_val);
			#sdev_ 	= np.std(Y_val);
			#Y_val 	= (Y_val - mu_) / sdev_;
		return Y_val,(signed_coeffs,f_ids)

	def osc(self,x,num_samples,parents_exist,pre_config):		
		Y_val=np.random.normal(0,np.pi/2.0,x.shape[0])
		dims = x.shape[1]
		f_ids = np.random.randint(4,6,dims)
		if pre_config is not None:
			f_ids = pre_config[1]
		tfs = []
		signed_coeffs = []
		if parents_exist:
			for i in range(dims):
				tfs.append(self.transform(x[:,i],f_ids[i]))

			dt = np.hstack(tfs)
			new_dims = dt.shape[1]
			s_dict={};
			s_dict[0] = -1
			s_dict[1] =  1

			if pre_config is None:
				coeffs = np.random.uniform(2,5,new_dims)
				signs  = np.array( [ s_dict[ss] for ss in list(np.random.randint(0,2,new_dims))  ]  )
				signed_coeffs = coeffs * signs
			else:
				signed_coeffs = pre_config[0]

			Y_val = np.dot(dt,signed_coeffs)+ np.random.normal(0,0.2,x.shape[0])
			#mu_ 	= np.mean(Y_val);
			#sdev_ 	= np.std(Y_val);
			#Y_val 	= (Y_val - mu_) / sdev_;
		return Y_val,(signed_coeffs,f_ids)


	# EXAMPLE def nn(self,..):
		

	
	def transform(self,s,function_id):
		zero_indices = np.where(s == 0)[0];
		negative_indices = np.where(s < 0)[0];
		eps = 0.0001;
		#dt = np.dtype('Float64');
		if function_id == 0:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
		elif function_id == 1:
			#x = np.zeros((s.shape[0],2))#,dtype=dt);
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			#x[:,0] = s.reshape(-1)**1;
			x[:,0] = s.reshape(-1)**2;
		elif function_id == 2:
			#x = np.zeros((s.shape[0],3))#,dtype=dt);
			x = np.zeros((s.shape[0], 1))
			#x[:,0] = s.reshape(-1)**1;
			#x[:,1] = s.reshape(-1)**2;
			x[:,2] = s.reshape(-1)**3;
		elif function_id == 3:
			x = np.zeros((s.shape[0],4))#,dtype=dt);
			x[:,0] = s.reshape(-1)**1;
			x[:,1] = s.reshape(-1)**2;
			x[:,2] = s.reshape(-1)**3;
			x[:,3] = s.reshape(-1)**4;
		elif function_id==4:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = np.sin(s).reshape(-1);
		elif function_id==5:
			x = np.zeros((s.shape[0],1))#,dtype=dt);
			x[:,0] = np.cos(s).reshape(-1);
		else:
			print ('WARNING: unknown function id',function_id, 'encountered, returning column as-is...');
			x = np.zeros((s.shape[0],1),dtype=dt);
			import ipdb; ipdb.set_trace();
			x[:,0] = x.reshape(-1)**1;
			
		return x;

	def normalize_data(self, data):
		new_data = np.zeros([data.shape[0], data.shape[1]])
		for n in range(data.shape[1]):
			vals = data[:,n]
			mu_ = np.mean(vals)
			sdev_ 	= np.std(vals)
			Y_val 	= (vals - mu_) / sdev_
			new_data[:,n] = Y_val
		return new_data


'''

	def intvp(self,x,num_samples,parents_exist,pre_config):
		print('pre config : ' + str(pre_config))
		#coeff = pre_config[0]
		coeff = np.array(pre_config[0])
		print('coeff : ' + str(coeff))
		coeff = coeff + 1
		# X should be fixed
		x = np.array(x)
		print('x sizes = ' + str(x.shape))
		noise = np.random.normal(0, 0.001, x.shape[0])
		print('noise : ' + str(len(noise)))
		print('X shape 0 : ' + str(x.shape[0]))
		# Y_val = np.array([c * x[i] for i, c in enumerate(coeff)])
		y_vals = coeff * x + noise
		print('Y-vals SIZE = ' + str(len(y_vals)))
		config = (coeff, pre_config[1])
		return y_vals, config
'''