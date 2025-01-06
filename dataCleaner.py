import numpy as np;

class DataCleaner:
	def __init__(self):
		self.x=2;
		
		
	def Clean(self,data,target,lim=3):
		return data,target;
		mu_ = np.mean(data,axis=0);
		sd_ = np.std(data,axis=0);
		
		upper_limit= mu_ + lim*sd_;
		lower_limit= mu_ - lim*sd_;
		
		z1 = data[:,]<= upper_limit 
		z2 = data[:,]>=lower_limit;
		
		#print upper_limit;
		#print lower_limit;
		k1 =[];
		for r in range(len(z1)):
			k1.append( z1[r].all() and z2[r].all());
		
		return data[np.where(k1)],target[np.where(k1)];
	
	'''def CleanMat(self,data,lim=3):
		#return data,target;
		mu_ = np.mean(data,axis=0);
		sd_ = np.std(data,axis=0);
		
		upper_limit= mu_ + lim*sd_;
		lower_limit= mu_ - lim*sd_;
		
		z1 = data[:,]<= upper_limit 
		z2 = data[:,]>=lower_limit;
		
		#print upper_limit;
		#print lower_limit;
		k1 =[];
		for r in range(len(z1)):
			k1.append( z1[r].all() and z2[r].all());
		
		return data[np.where(k1)];'''

	def CleanMat(self, data, lim=3):
		mu_ = np.mean(data, axis=0)
		sd_ = np.std(data, axis=0)

		# Calculate the upper and lower limits for each column
		upper_limit = mu_ + lim * sd_
		lower_limit = mu_ - lim * sd_

		# Iterate through the data and replace out-of-bounds values with the mean
		for i in range(data.shape[0]):  # Iterate over rows
			for j in range(data.shape[1]):  # Iterate over columns
				if data[i, j] > upper_limit[j] or data[i, j] < lower_limit[j]:
					data[i, j] = mu_[j]  # Replace with column mean

		return data
			
'''
def main():
	dc = DataCleaner();
	
	data= np.array([[1,2,3],[1,4,0],[-100,2,-1],[0,5,0]]);
	
	print dc.Clean(data,1);
	
	
main();
'''
