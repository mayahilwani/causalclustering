from globeWrapper import GlobeWrapper;
from spotWrapper import SpotWrapper;
from slope import Slope;
import sys;

def main():
	filepath= "C:/Users/ziadh/Documents/CausalGen-Osman/periodic/test_2to5_3"
	slope = Slope()
	spot = SpotWrapper(slope)
	n = 50
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
