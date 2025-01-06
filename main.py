from globeWrapper import GlobeWrapper;
from spotWrapper import SpotWrapper;
from slope import Slope;
import sys;

def main():
	filename= "test_data2/expirement" #"shift/experiment5"
	slope = Slope()
	spot = SpotWrapper(slope)
	n = 10
	for i in range(n):
		filename1 = filename + str(i+1)
		print("FILE NAME " + str(filename1))
		spot.idk(filename1)


'''
#Max_Interactions=2;	#See the Instantiation section of the publication
	#log_results=True;	#Set this to true if you would like to store the log of the experiment to a text file
	#verbose=True;	#Set this to true if you would like see the log output printed to the screen
	#globe= GlobeWrapper(Max_Interactions,log_results,verbose);
	if(result):
		print('SPLIT');
	else:
		print('DO NOT SPLIT');'''

main();
