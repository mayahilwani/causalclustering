'''
This file is unfinnished. POTENTIAL OF REMOVAL
'''
import numpy as np;
from combinator import Combinator;
from edge import Edge;
#from sampler import Sampler;
from dataTransformer import DataTransformer
from spot import Spot;
from node import Node;

class SpotWrapper:
    def __init__(self, slp, dims=0, M=2):
        self.slope_ = slp;
        #self.sampler = Sampler();
        self.Transformer = DataTransformer(True);
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.M = M;
        print("Max interactions set to degree ", self.M);

    def idk(self, filename):
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen

        spot = Spot(Max_Interactions, log_results, verbose);

        spot.loadData(filename);

        found_intv, labels_array = spot.run();
        try:
            base_path = "C:/Users/ziadh/Documents/CausalGen-Osman"
            file = base_path + "/" + filename;
            intv_file = f"{file}/interventions1.txt"
            intvs = np.loadtxt(intv_file, delimiter=',')

        except Exception as e:
            print(f"An error occurred: {e}")
        intv_cnt = 0
        cnt = 0
        for intv in intvs:
            intv_cnt += 1
            if intv in found_intv:
                cnt += 1
        print(str(cnt) + " INTERVENTIONS FOUND OUT OF " + str(intv_cnt) + " INTERVENTIONS")
        if cnt == intv_cnt:
            print("ALL FOUND !!!")

        #spot.analyzeLabels()
