from spotRandom import SpotRandom;
from spot import Spot;
from slope import Slope;
import sys;

def main():
    filepath= "./data_gen/uniform/non_linear_2/test_2to5_0/"
    slope = Slope()
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    spot = Spot(Max_Interactions, log_results, verbose);
    filename1 = filepath + str('expirement1')
    spot.loadData(filename1)
    spot.run([7])



main();
