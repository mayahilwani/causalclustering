
import numpy as np
import pandas as pd
import networkx as nx

def generate_mixture_sachs(fpath, verbose=False):
    """ source: https://github.com/BigBang0072/mixture_mec/blob/comp_sel/scm_module.py """
    # Reading the full dataset
    df = pd.read_csv(fpath, delimiter="\t")

    # Genetaing the samples from each of the mixture
    intv_args_dict = {}
    mixture_samples = []
    intv_targets = [("Akt", 3), ("PKC", 4), ("PIP2", 5), ("Mek", 6), ("PIP3", 7)]
    # Getting the names of the variables
    var_names = df.drop(columns=["experiment"]).columns.tolist()
    var2idx_dict = {var: idx for idx, var in enumerate(var_names)}
    idx2var_dict = {val: key for key, val in var2idx_dict.items()}
    # Getting the adjacecny matrix for this dataset
    A = get_sachs_adj_matrix(var2idx_dict)
    num_nodes = A.shape[0]

    # We will keep the net number of sample same so that the number of component doesnt have any effect
    # num_samples = num_samples//(len(intv_targets)+1)

    # Adding the observational data
    if verbose: print("Observational data")
    intv_args_dict["obs"] = {}
    intv_args_dict["obs"]["tgt_idx"] = None
    obs_samples = df[(df["experiment"] == 1) | (df["experiment"] == 2)
                     ].drop(columns=["experiment"]).to_numpy()
    if verbose: print("num samples:", obs_samples.shape[0])
    mixture_samples.append(obs_samples)
    intv_args_dict["obs"]["samples"] = obs_samples
    intv_args_dict["obs"]["true_params"] = dict(
        Si=np.cov(obs_samples, rowvar=False),
        mui=np.mean(obs_samples, axis=0),
        Ai=A,
    )

    # Now one by one we will add the internvetional data that we know of
    for tgt, expt_num in intv_targets:
        if verbose: print("Internvetional data for:", tgt, f"({var2idx_dict[tgt]})")
        # Getting the internvetional data for this target
        intv_samples = df[df["experiment"] == expt_num].drop(
            columns=["experiment"]).to_numpy()
        if verbose: print("num_samples for {}: {}".format(tgt, intv_samples.shape[0]))
        mixture_samples.append(intv_samples)
        # Addig the internvetion info
        intv_args_dict[tgt] = {}
        intv_args_dict[tgt]["tgt_idx"] = var2idx_dict[tgt]
        # This will have clean mixture samples
        intv_args_dict[tgt]["samples"] = intv_samples   # this should be the intervention samples.

        # Getting the new adjancecy matrix for this intervened dist (do intv)
        Ai = A.copy()
        Ai[var2idx_dict[tgt], :] = 0.0
        intv_args_dict[tgt]["true_params"] = dict(
            Si=np.cov(intv_samples, rowvar=False),
            mui=np.mean(intv_samples, axis=0),
            Ai=Ai,
        )

    # Acculmulating the samples in to one big matrix
    mixture_samples = np.concatenate(mixture_samples, axis=0)
    if verbose: print("Total number of samples: ", mixture_samples.shape[0])

    return intv_args_dict, mixture_samples, num_nodes, idx2var_dict, A


def get_sachs_adj_matrix(var2idx_dict):
    """ source: https://github.com/BigBang0072/mixture_mec/blob/comp_sel/scm_module.py """
    num_nodes = len(var2idx_dict)
    A = np.zeros((num_nodes, num_nodes))
    A[var2idx_dict["Akt"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Erk"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Raf"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["JNK"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["p38"], var2idx_dict["PKA"]] = 1.0
    A[var2idx_dict["Akt"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PIP2"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PLCg"], var2idx_dict["PIP3"]] = 1.0
    A[var2idx_dict["PKC"], var2idx_dict["PIP2"]] = 1.0
    A[var2idx_dict["PIP2"], var2idx_dict["PLCg"]] = 1.0
    A[var2idx_dict["PKC"], var2idx_dict["PLCg"]] = 1.0
    A[var2idx_dict["Erk"], var2idx_dict["Mek"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["Raf"]] = 1.0
    A[var2idx_dict["Mek"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["Raf"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["JNK"], var2idx_dict["PKC"]] = 1.0
    A[var2idx_dict["p38"], var2idx_dict["PKC"]] = 1.0

    return A

if __name__ == "__main__":
    verbose = True
    file_path = "sachs_yuhaow.csv"
    df = pd.read_csv("sachs_yuhaow.csv", delimiter="\t")
    df_no_experiment = df.drop(columns=["experiment"])
    print(f"shape: {df_no_experiment.shape}")
    print(f"df.head(): {df_no_experiment.head()}")
    #df_no_experiment.to_csv("data2.txt", sep=",", index=False)
    df.to_csv("full_data.txt", sep=",", index=False, header=False)
    intv_args_dict, mixture_samples, num_nodes, idx2var_dict, true_A = generate_mixture_sachs(file_path, verbose)
    true_graph = nx.from_numpy_array(true_A.T, create_using=nx.DiGraph)
    #np.savetxt("truth1.txt", true_A, fmt="%d", delimiter=",")

    if verbose:  print("Causal Edges:\n" + ", ".join([f"{idx2var_dict[e[0]]}->{idx2var_dict[e[1]]}" for e in true_graph.edges]))


