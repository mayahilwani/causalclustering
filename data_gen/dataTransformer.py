import numpy as np;
import random
from sklearn.ensemble import IsolationForest

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

	def intv_f(self, x, num_samples, parents_exist, strength=1, noise_level=0, pre_config=None):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]

		# Determine function type and intervention parameters
		func_type = self.determine_function_type(f_ids)
		shift, coef_scale = self.get_intervention_parameters(func_type, strength)

		#print(f'Function Type: {func_type}')
		#print(f'Intervention Strength: {strength} -> Shift: {shift}, Coef Scale: {coef_scale}')
		#print(f'Noise Level: {noise_level} -> Noise Std: {noise_scale}')

		# Apply selective flipping and scaling
		new_coeff = coeff.copy()
		sign = -1  # if flip else 1
		if strength == 4:
			for i in range(dims):
				new_coeff[i] *= sign * coef_scale * np.random.uniform(0.9, 1.2)
		else:
			intv_pa = random.choice(range(dims))
			new_coeff[intv_pa] *= sign * coef_scale * np.random.uniform(0.9, 1.1)
		# Transform input
		tfs = [self.transform(x[:, i], f_ids[i]) for i in range(dims)]
		dt = np.hstack(tfs)

		Y_val = np.dot(dt, new_coeff)
		noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
		noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
		Y_val = Y_val + noise
		#Y_val = np.dot(dt, new_coeff).reshape(-1, 1) + shift + noise

		new_config = (new_coeff, f_ids)
		return Y_val, new_config


	def intv_s(self, x, num_samples, parents_exist, strength=1, noise_level=0, pre_config=None):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]

		func_type = self.determine_function_type(f_ids)
		shift, coef_scale = self.get_intervention_parameters(func_type, strength)
		coef_scale = coef_scale + np.random.uniform(0.1, 0.4)
		#print(f'Function Type: {func_type}')
		#print(f'Scaling Strength: {strength} -> Coef Scale: {coef_scale}')
		#print(f'Noise Level: {noise_level} -> Noise Std: {noise_scale}')

		new_coeff = coeff.copy()
		if strength == 3:
			for i in range(dims):
				new_coeff[i] *= coef_scale * np.random.uniform(0.9, 1.2)
		else:
			intv_pa = random.choice(range(dims))
			print(type(new_coeff[intv_pa]), new_coeff[intv_pa])
			print(coef_scale * np.random.uniform(0.9, 1.1))

			new_coeff[intv_pa] *= coef_scale * np.random.uniform(0.9, 1.1)
		tfs = [self.transform(x[:, i], f_ids[i]) for i in range(dims)]
		dt = np.hstack(tfs)
		Y_val = np.dot(dt, new_coeff)
		noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
		noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
		Y_val = Y_val + noise
		#Y_val = np.dot(dt, new_coeff).reshape(-1, 1) + noise

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

	def intv_sh(self, x, num_samples, parents_exist, strength=1, noise_level=0, pre_config=None):
		coeff = np.array(pre_config[0])
		f_ids = pre_config[1]
		dims = x.shape[1]

		func_type = self.determine_function_type(f_ids)
		shift, _ = self.get_intervention_parameters(func_type, strength)
		noise_scale = self.get_noise_scale(noise_level)

		#print(f'Function Type: {func_type}')
		#print(f'Shift Strength: {strength} -> Shift: {shift}')
		#print(f'Noise Level: {noise_level} -> Noise Std: {noise_scale}')

		new_coeff = coeff.copy()  # No scaling, just a shift

		tfs = [self.transform(x[:, i], f_ids[i]) for i in range(dims)]
		dt = np.hstack(tfs)
		Y_val = np.dot(dt, new_coeff)
		noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
		noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
		Y_val = Y_val + shift + noise
		#Y_val = np.dot(dt, new_coeff).reshape(-1, 1) + shift + noise

		new_config = (new_coeff, f_ids)
		return Y_val, new_config

	def determine_function_type(self, f_ids):
		f_ids_set = set(f_ids)
		if f_ids_set.issubset({0}):
			return "linear"
		elif f_ids_set.issubset({4, 5}):
			return "periodic"
		elif any(f in {1, 2, 3} for f in f_ids):
			return "polynomial"
		else:
			return "mixed"  # fallback
    # The data looks too noisy.
	def get_noise_scale(self, noise_level):
		levels = {
			0: 0.05,
			1: 0.09,
			2: 0.18,
		}
		#print(f'Noise Level: {levels.get(noise_level, 0.1)}')
		return levels.get(noise_level, 0.1)

	def get_intervention_parameters(self, func_type, strength):
		# Dummy implementation — adjust to your needs
		shift = max((strength * 0.5), 0.3) #0.1
		coef_scale = 1 + (strength * 0.5) #0.05
		return shift, coef_scale
		# return shift, coef_scale

	def lin(self,x,num_samples,parents_exist, intv_strength=1, noise_level=0, pre_config=None):
		noise_scale = self.get_noise_scale(noise_level) # * np.sqrt(x.shape[1])
		Y_val = np.random.normal(0, noise_scale, x.shape[0])
		#Y_val=np.random.normal(0,3*x.shape[1],x.shape[0])
		#Y_val = np.random.uniform(low=-3*x.shape[1], high=3*x.shape[1], size=x.shape[0])
		# Apply Isolation Forest to detect and remove outliers from Y_val
		'''iso_forest = IsolationForest(contamination=0.05, random_state=42)
		outliers = iso_forest.fit_predict(Y_val.reshape(-1, 1))
		inlier_values = Y_val[outliers == 1]
		Y_val_fixed = Y_val.copy()
		outlier_indices = np.where(outliers == -1)[0]
		Y_val_fixed[outlier_indices] = np.random.choice(inlier_values, size=len(outlier_indices), replace=True)
		Y_val = Y_val_fixed'''
		dims = x.shape[1]
		#f_ids = np.random.randint(0,2,dims) # was 3
		f_ids = np.zeros(dims, dtype=int)
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

			Y_val = np.dot(dt, signed_coeffs)
			noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
			noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
			Y_val = Y_val + noise
			#Y_val 	= np.dot(dt,signed_coeffs)+np.random.normal(0,2*x.shape[1],x.shape[0])
			#mu_ 	= np.mean(Y_val);
			#sdev_ 	= np.std(Y_val);
			#Y_val 	= (Y_val - mu_) / sdev_;

		return Y_val, (signed_coeffs, f_ids)

	def poly(self,x,num_samples,parents_exist, intv_strength=1, noise_level=0, pre_config=None):
		noise_scale = self.get_noise_scale(noise_level) * np.sqrt(x.shape[1])
		Y_val = np.random.normal(0, noise_scale, x.shape[0])
		#Y_val=np.random.normal(0,3*x.shape[1],x.shape[0])
		# Apply Isolation Forest to detect and remove outliers from Y_val
		'''iso_forest = IsolationForest(contamination=0.05, random_state=42)
		outliers = iso_forest.fit_predict(Y_val.reshape(-1, 1))
		inlier_values = Y_val[outliers == 1]
		Y_val_fixed = Y_val.copy()
		outlier_indices = np.where(outliers == -1)[0]
		Y_val_fixed[outlier_indices] = np.random.choice(inlier_values, size=len(outlier_indices), replace=True)
		Y_val = Y_val_fixed'''
		dims = x.shape[1]
		# Generate until at least one element is 1
		while True:
			f_ids = np.random.randint(0, 2, dims)
			if np.any(f_ids == 1):  # Check if there's at least one 1
				break
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

			Y_val = np.dot(dt, signed_coeffs)
			noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
			noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
			Y_val = Y_val + noise
			
			#print(Y_val.shape)
			#Y_val 	= np.dot(dt,signed_coeffs)+np.random.normal(0,2*x.shape[1],x.shape[0])
			#mu_ 	= np.mean(Y_val);
			#sdev_ 	= np.std(Y_val);
			#Y_val 	= (Y_val - mu_) / sdev_;

		return Y_val,(signed_coeffs,f_ids)

	def osc(self,x,num_samples,parents_exist, intv_strength=1, noise_level=0, pre_config = None):
		noise_scale = self.get_noise_scale(noise_level) #* np.sqrt(x.shape[1])
		Y_val = np.random.normal(0, noise_scale, x.shape[0])
		#Y_val=np.random.normal(0,np.pi/2.0,x.shape[0])
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

			Y_val = np.dot(dt, signed_coeffs)
			noise_scale = self.get_noise_scale(noise_level) * np.std(Y_val)
			noise = np.random.normal(0, noise_scale, size=(x.shape[0]))
			Y_val = Y_val + noise
			#Y_val = np.dot(dt,signed_coeffs)+ np.random.normal(0,0.2,x.shape[0])
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
