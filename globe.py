import numpy as np;

from combinator import Combinator;
from edge import Edge;
#from sampler import Sampler;
from dataTransformer import DataTransformer
from node import Node;
from dataCleaner import DataCleaner;


class Globe:

    def __init__(self, slp, dims=0, M=2):
        self.slope_ = slp;
        #self.sampler = Sampler();
        self.Transformer = DataTransformer(True);
        self.terms = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1, 8: 1}
        self.F = 9;
        self.V = dims;
        self.M = M;
        print("Max interactions set to degree ", self.M);

    def GetEdgeAdditionCost(self, parents, child, labels = None):
        if labels is None:
            best_combination, best_compression, best_bits, best_coeffs, best_absolute = self.evaluate_function_combinations(
                parents, child, labels, []
                )
        # Evaluate the best function combination for the parents-child relationship
        else:
            best_combination1, best_compression1, best_bits1, best_coeffs1, best_absolute1, best_combination2, best_compression2, best_bits2, best_coeffs2, best_absolute2 = (
                self.evaluate_function_combinations(
                parents, child, labels, []
                ))

        '''# Print the results
        print("Best function combination:", best_combination)
        print("Best compression gain in bits:", best_compression)
        print("New bits required:", best_bits)
        print("Best coefficients:", best_coeffs)
        print("Absolute gain in bits:", best_absolute)'''
        if labels is None:
            return best_compression, best_bits, best_absolute, best_combination, best_coeffs;
        else:
            return best_combination1, best_compression1, best_bits1, best_coeffs1, best_absolute1, best_combination2, best_compression2, best_bits2, best_coeffs2, best_absolute2;

    # Function to evaluate multiple function combinations for edges
    def evaluate_function_combinations(self, parents, child, labels,  edge_parents_child):
        if labels is None:
            best_combination = None
            best_compression = float('inf')  # Start with a high number
            best_coeffs = None
            best_absolute = None
            best_bits = None

            # Iterate over all possible combinations of functions for the edges
            for f_combination in self.generate_all_combinations(self.F, len(parents)):
                # Create edges with the selected function IDs for each parent-child relationship
                new_edges = [Edge(f_id, [], [], 0) for f_id in f_combination]

                # Evaluate the cost of this specific combination of function IDs
                gain_in_bits, new_bits, coeffs, absolute_bits = self.GetCombinationCost(parents, new_edges, child)
                # If this combination results in better compression, update the best found
                if gain_in_bits < best_compression:
                    best_combination = f_combination
                    best_compression = gain_in_bits
                    best_coeffs = coeffs
                    best_absolute = absolute_bits
                    best_bits = new_bits
            return best_combination, best_compression, best_bits, best_coeffs, best_absolute
        else:
            best_combination1 = None
            best_compression1= float('inf')
            best_coeffs1 = None
            best_absolute1 = None
            best_bits1 = None
            best_combination2 = None
            best_compression2 = float('inf')
            best_coeffs2 = None
            best_absolute2 = None
            best_bits2 = None

            # Iterate over all possible combinations of functions for the edges
            for f_combination in self.generate_all_combinations(self.F, len(parents)):
                # Create edges with the selected function IDs for each parent-child relationship
                new_edges = [Edge(f_id, [], [], 0) for f_id in f_combination]

                # Evaluate the cost of this specific combination of function IDs on first group
                parents1 = []
                for parent in parents:
                    parents1.append(Node(parent.GetData()[labels == 0]), self)
                child1 = Node(child.GetData()[labels == 0], self)
                gain_in_bits1, new_bits1, coeffs1, absolute_bits1 = self.GetCombinationCost(parents1, new_edges, child1)
                # If this combination results in better compression, update the best found
                if gain_in_bits1 < best_compression1:
                    best_combination1 = f_combination
                    best_compression1 = gain_in_bits1
                    best_coeffs1 = coeffs1
                    best_absolute1 = absolute_bits1
                    best_bits1 = new_bits1
                # Evaluate the cost of this specific combination of function IDs on first group
                parents2 = []
                for parent in parents:
                    parents2.append(Node(parent.GetData()[labels == 1]), self)
                child2 = Node(child.GetData()[labels == 1], self)
                gain_in_bits2, new_bits2, coeffs2, absolute_bits2 = self.GetCombinationCost(parents2, new_edges,
                                                                                            child2)
                # If this combination results in better compression, update the best found
                if gain_in_bits2 < best_compression2:
                    best_combination2 = f_combination
                    best_compression2 = gain_in_bits2
                    best_coeffs2 = coeffs2
                    best_absolute2 = absolute_bits2
                    best_bits2 = new_bits2

            best_combination1, best_compression1, best_bits1, best_coeffs1, best_absolute1, best_combination2, best_compression2, best_bits2, best_coeffs2, best_absolute2

    # Function to generate all combinations of functions for a given number of parents
    def generate_all_combinations(F, num_parents):
        from itertools import product
        # Generate all possible combinations of function IDs (0 to F-1) for each parent node
        return list(product(range(F), repeat=num_parents))

    '''def GetAverageCompression(self, parents1, parent2, child, edge_parents1_child, edge_parent2_child, max_iter=100):
        rows = child.GetData().shape[0]
        gains = np.zeros((max_iter, self.F));
        parent_count = np.array([len(parents1)]) + 1;
        dt = [];
        dt.append(child.GetData().reshape(rows, 1) ** 0);
        for i in range(len(parents1)):
            dt.append(self.Transformer.TransformData(parents1[i].GetData(), edge_parents1_child[i].GetFunctionId()));

        running_average = 0;
        thresh = 10;
        tolerance = 0;
        for i in range(max_iter):
            mutated_data = self.sampler.Mutate(parent2.GetData());
            for fid in range(self.F):
                dt2 = dt;
                app_var = self.Transformer.TransformData(mutated_data, fid);
                dt2.append(app_var);

                source = np.hstack(dt2);
                target = child.GetData();

                new_bits, coeffs = self.ComputeScore(source, target, rows, child.GetMinDiff(), parent_count);
                gains[i, fid] = max(0, child.GetCurrentBits() - new_bits);
                del dt2[-1];

            if np.abs(running_average - np.mean(gains)) < thresh:
                tolerance = tolerance + 1;
                if tolerance > 200:
                    break;
            else:
                tolerance = 0;
            running_average = np.mean(gains);
        return np.mean(gains);
'''
    def GetCombinationCost(self, parents, edge_parents_child, child, labels, debug=False):

        dt = [];
        parent_count = np.array([len(parents)]);

        rows = child.GetData().shape[0]
        dt.append(child.GetData().reshape(rows, -1) ** 0);
        for i in range(len(parents)):
            dt.append(self.Transformer.TransformData(parents[i].GetData(), edge_parents_child[i].GetFunctionId()));

        source = np.hstack(dt);
        target = child.GetData();

        new_bits, coeff = self.ComputeScore(source, target, rows, child.GetMinDiff(), parent_count);
        gain_in_bits = new_bits / child.GetCurrentBits();
        absolute_gain_in_bits = max(0, child.GetCurrentBits() - new_bits[0]);

        return gain_in_bits, new_bits, coeff, np.array([absolute_gain_in_bits]);

    def ComputeScore(self, source, target, rows, mindiff, k, show_graph=False):
        base_cost = self.slope_.model_score(k) + k * np.log2(self.V);
        sse, model, coeffs, hinges, interactions, rearth= self.slope_.FitSpline(source, target, self.M, show_graph);
        base_cost = base_cost + self.slope_.model_score(hinges) + self.AggregateHinges(interactions, k);
        cost = self.slope_.gaussian_score_emp_sse(sse, rows, mindiff) + model + base_cost;
        return cost, coeffs;


    def AggregateHinges(self, hinges, k):
        cost = 0;
        flag = 1;

        for M in hinges:
            cost += self.slope_.logN(M) + Combinator(M, k) + M * np.log2(self.F);
        return cost;

