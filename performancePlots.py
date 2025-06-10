import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# --- Setup Paths ---
#parent_dir = "./tests/test_50_2500_500_0_2_2_3_0_0_0_0/"    #test_50_2000_500_3_0_2_3_0_0_1_0  test_50_2000_500_3_1_2_3_0_0_1_2   test_50_2000_500_3_2_2_3_0_0_1_2
parent_dir = "/Users/mayahilwani/PycharmProjects/msc-mhilwani/tests/test_50_2250_250_3_0_2_3_0_0_3_2/"    #test_50_2000_500_3_0_2_3_0_0_1_0  test_50_2000_500_3_1_2_3_0_0_1_2   test_50_2000_500_3_2_2_3_0_0_1_2
experiment_folders = sorted(glob.glob(os.path.join(parent_dir, "experiment*")))
experiment_folders = [folder.replace("\\", "/") for folder in experiment_folders]

# --- Read and Combine Data ---
global_data = []

for folder in experiment_folders:
    file_path = os.path.join(folder, "node_STATS.txt").replace("\\", "/")
    if not os.path.exists(file_path):
        continue

    local_df = pd.read_csv(file_path, header=0)
    local_df.columns = local_df.columns.str.strip()
    global_data.append(local_df)

# Combine all into one DataFrame
df = pd.concat(global_data, ignore_index=True)

#print(f" last global id : {next_global_id}")
#global_df = pd.DataFrame(global_data)
# --- ARI Score Boxplot (k=2 & true_score_diff <= 0) ---
low_score_df = df[
    (df["k"] == 2) & (df["true_split"] != 0) &
    (pd.to_numeric(df["true_score_diff"], errors="coerce") <= 0)
]
#low_score = low_score_df[low_score_df["true_split"] == 1]
all_scores = df[(df["true_split"] == 1) & (df["k"] == 2)]
print(f"LOW SCORE CASES: {low_score_df.shape[0]} . ALL CASES: {all_scores.shape[0]}")

#global_df = global_df[(global_df["k"] == 2) & (pd.to_numeric(global_df["true_score_diff"], errors="coerce") > 300)]

# --- Type Conversions ---
numeric_cols = [
    "true_split", "found_split", "cc_ari", "gmm_ari","gmm_ari_res", "kmeans_ari","kmeans_ari_res", "spectral_ari", "spectral_ari_res",
    "cc_nmi", "gmm_nmi","gmm_nmi_res", "kmeans_nmi","kmeans_nmi_res", "spectral_nmi","spectral_nmi_res",
    "cc_fmi", "gmm_fmi","gmm_fmi_res", "kmeans_fmi","kmeans_fmi_res", "spectral_fmi", "spectral_fmi_res"
]
numeric_cols.extend(["initial_split", "per_intv", "parent_intv"])

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
# Filter the TP cases
tp_df = df[(df["true_split"] == 1) & (df["found_split"] == 1)]

# Count number of TP cases per num_parents
tp_counts = tp_df["num_parents"].value_counts().sort_index()

print("Number of TP cases per num_parents:")
for num_parents, count in tp_counts.items():
    print(f"  num_parents = {num_parents} → {count} cases")



# --- ARI Score Boxplot (TP only) ---
filtered_df = df[(df["true_split"] == 1) & (df["found_split"] == 1) & (df["num_parents"].isin([1, 2, 3, 4]))]
#ari_cols = ["cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"]
nmi_cols = ["cc_nmi", "gmm_nmi","gmm_nmi_res", "kmeans_nmi",  "kmeans_nmi_res", "spectral_nmi", "spectral_nmi_res"]

'''tp_1 = tp_df[tp_df["num_parents"] == 1]
print("Non-NaN values for num_parents = 1:")
print(tp_1[nmi_cols].notna().sum())'''

# Convert to long format
melted_df = pd.melt(
    filtered_df,
    id_vars=["num_parents"],
    value_vars=nmi_cols,
    var_name="method",
    value_name="score"
)

plt.figure(figsize=(8, 6))
#sns.boxplot(data=filtered_df[ari_cols])
sns.boxplot(data=melted_df,
    x="method",
    y="score",
    hue="num_parents")
#plt.title("ARI Score Comparison (True & Found Split)")
plt.title("NMI Score Comparison for TP")
plt.xticks(rotation=15)
#plt.ylabel("ARI Score")
plt.ylabel("NMI Score")
plt.xlabel("Methods")
plt.tight_layout()
plt.show()
print('First box plot with all TP cases')
#plt.close()

# Convert ARI columns to numeric if not already
#low_score_df[ari_cols] = low_score_df[ari_cols].apply(pd.to_numeric, errors="coerce")
low_score_df[nmi_cols] = low_score_df[nmi_cols].apply(pd.to_numeric, errors="coerce")

# Check if we have any data to plot
if not low_score_df.empty:
    plt.figure(figsize=(8, 6))
    #sns.boxplot(data=low_score_df[ari_cols])
    sns.boxplot(data=low_score_df[nmi_cols])
    #plt.title("ARI Score Comparison (k=2 & true_score_diff ≤ 0)")
    plt.title("NMI Score Comparison (k=2 & true_score_diff ≤ 0)")
    plt.xticks(rotation=15)
    #plt.ylabel("ARI Score")
    plt.ylabel("NMI Score")
    plt.xlabel("Methods")
    plt.tight_layout()
    plt.show()
    print('Second box plot with negative true score cases')
else:
    print("⚠️ No data to plot for k=2 & true_score_diff ≤ 0.")

# Identify nodes that were already counted in TP or FN
#used_nodes = set(global_df[(global_df["true_split"] == 1)]["global_node_id"])
#print(f"Nodes with true split (used for TP/FN): {len(used_nodes)}")

# Count TP and FN as usual
TP = df[(df["true_split"] == 1) & (df["found_split"] == 1)].shape[0]
FN = df[(df["true_split"] == 1) & (df["found_split"] == 0)].shape[0]
print(f"True split {df[(df["true_split"]==1)].shape[0]}  and TP = {TP}  FN = {FN}")

# FP and TN
FP = df[(df["k"] == 2) & (df["true_split"] == 0) & (df["found_split"] == 1)].shape[0]
TN = df[(df["k"] == 2) & (df["true_split"] == 0) & (df["found_split"] == 0)].shape[0]
print(f"NOT true split {df[(df["true_split"]==0) & (df["k"] ==2)].shape[0]}  and FP = {FP}  TN = {TN}")


# --- Final Summary ---
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
