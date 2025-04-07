from cc import CC
from slope import Slope;
import sys;

def main():
    filepath= "./tests/test_10_2000_1000_0_1_2_2_0/"
    slope = Slope()
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    cc = CC(Max_Interactions, log_results, verbose);
    filename1 = filepath + str('experiment7')
    cc.loadData(filename1)
    cc.run(2, [9], False, False)



main();
