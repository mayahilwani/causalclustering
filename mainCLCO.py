import numpy as np;
from clusterConsistency import ClusterConsistency
from combinator import Combinator;
from edge import Edge;
#from sampler import Sampler;
from dataTransformer import DataTransformer
from cc import CC
from node import Node;
import time
import RFunctions as rf
import multiprocessing


def idk(filename):
    Max_Interactions = 2;  # See the Instantiation section of the publication
    log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
    verbose = True;  # Set this to true if you would like see the log output printed to the screen
    cco = ClusterConsistency(Max_Interactions, log_results, verbose);
    cco.loadData(filename);
    cco.run()
# test_50_2000_500_0_1_2_4_0_0/experiment36/ 8 7 1

'''
Cost of ONE MODEL : [56872.35063107]
COST SPLIT [56598.40635899]
COST OLD : [56598.40635899]
COST NEW : [55867.67242269]
ARI OLD : 0.5935789230219392
ARI NEW : 0.9458863562299961
'''

def main():
    filename = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/test_50_2000_500_0_1_2_4_0_0/experiment16/"
    idk(filename);

if __name__ == "__main__":
    main()


