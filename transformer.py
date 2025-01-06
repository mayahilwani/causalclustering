'''
NOT USED
'''
def intv(self, x, num_samples, parents_exist, pre_config):
		coeff = np.array(pre_config[0])
		print('OLD COEF: ' + str(coeff))
        # Apply a transformation to the coefficients
		#new_coeff = np.random.normal(0, np.pi / 2.0, len(coeff))
		new_coeff = coeff
		print('NEW COEF: ' + str(new_coeff))
		#y_vals = np.random.normal(0, np.pi / 2.0, x.shape[0])
		f_ids = pre_config[1]
		tfs = []
		if parents_exist:
			for i in range(x.shape[1]):
				tfs.append(self.transform(x[:, i], f_ids[i]))
			dt = np.hstack(tfs)
			shift = 0
			y_vals = np.dot(dt, new_coeff) + np.random.normal(0, 0.2, x.shape[0]) + shift
			#y_vals = np.random.normal(0, np.pi / 2.0, x.shape[0])
			#mu_ = np.mean(y_vals)
			#sdev_ = np.std(y_vals)
			#y_vals = (y_vals - mu_) / sdev_
		return y_vals, (new_coeff, f_ids)