from cc import CC
from slope import Slope;
import sys;

def main():
    filepath= "./tests/test_10_1000_500_0_0_2_2_0/"
    slope = Slope()
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    cc = CC(Max_Interactions, log_results, verbose);
    filename1 = filepath + str('experiment6')
    cc.loadData(filename1)
    cc.run(2, [7], False, False)



main();
