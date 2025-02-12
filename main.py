from globeWrapper import GlobeWrapper;
from spotWrapperRandom import SpotWrapperRandom;
from slope import Slope;
import sys;

def main():
	filepath= "C:/Users/ziadh/Documents/CausalGen-Osman/norm/non_linear_2/test_2to5_0_2"
	slope = Slope()
	spot = SpotWrapperRandom(slope)
	n = 5
	spot.generate_stats(filepath, n)
	#filename1 = file_name + str(1)
	#spot.idk(filename1)


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
