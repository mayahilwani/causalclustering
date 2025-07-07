from cc import CC
from slope import Slope;
import sys;
# test with M M where my method is better test_50_2000_500_3_2_2_3_0_0_1_2 experiment1 8
# test with 60-40 bad for all  test_50_1500_1000_0_2_2_3_0_0_0_1 experiment1 6
# test with polynomial scale in 3D  test_50_2000_500_1_1_2_3_0_0_2_1 experiment1 3
# test with polynomial scale 2D test_50_2000_500_1_1_2_3_0_0_2_1 experiment10 8
# test with polynomial shift 2D test_50_2000_500_3_1_2_3_0_0_2_1 experiment4 8
# test with periodic flip where CluC outperforms ALL test_50_2000_500_0_2_2_3_0_0_0_1  experiment22 9
# test with polynomial flip test_50_2000_500_0_1_2_3_0_0_0_1  experiment7  3
def main():
    #filepath= "./tests/test_50_2000_500_1_2_2_4_0_0/"
    filepath = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/test_50_2000_500_1_1_2_3_0_0_2_1/" #/msc-mhilwani # test_1_2250_250_0_1_2_2_0_0_0_1
    slope = Slope()
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    cc = CC(Max_Interactions, log_results, verbose);
    filename1 = filepath + str('experiment15')
    cc.loadData(filename1)
    cc.run(2, [7], False, False)

# good example of strong intervention but yet better test_50_2000_500_3_1_2_3_0_0_2_2 experiment2 [6,2]
# example first one FN (neg true score ) second correct
# test_50_2000_500_3_1_2_3_0_0_1_2 experiment2 [7,6]
# example test_50_2000_500_1_0_2_3_0_0_2_1 experiment1 [9]

main();
