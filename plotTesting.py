from cc import CC
from slope import Slope;
import sys;

def main():
    #filepath= "./tests/test_50_2000_500_1_2_2_4_0_0/"
    filepath = "/Users/mayahilwani/PycharmProjects/tests/test_1_2250_250_1_2_2_2_0_0_2_1/" #/msc-mhilwani # test_1_2250_250_0_1_2_2_0_0_0_1
    slope = Slope()
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    cc = CC(Max_Interactions, log_results, verbose);
    filename1 = filepath + str('experiment1')
    cc.loadData(filename1)
    cc.run(2, [5,6,4,8,3], False, False)

# good example of strong intervention but yet better test_50_2000_500_3_1_2_3_0_0_2_2 experiment2 [6,2]
# example first one FN (neg true score ) second correct  test_50_2000_500_3_1_2_3_0_0_1_2 experiment2 [7,6]
# example test_50_2000_500_1_0_2_3_0_0_2_1 experiment1 [9]

main();
