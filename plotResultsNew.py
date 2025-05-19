import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# --- Setup Paths ---
parent_dir = "./tests/test_50_2250_250_3_1_2_3_0_0_2_1/"    #test_50_2000_500_3_0_2_3_0_0_1_0  test_50_2000_500_3_1_2_3_0_0_1_2   test_50_2000_500_3_2_2_3_0_0_1_2
experiment_folders = sorted(glob.glob(os.path.join(parent_dir, "experiment*")))
experiment_folders = [folder.replace("\\", "/") for folder in experiment_folders]

# --- Read and Combine Data ---
global_data = []
node_id_mapping = {}
next_global_id = 0

for folder in experiment_folders:
    file_path = os.path.join(folder, "node_STATS.txt").replace("\\", "/")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        node_key = (folder, row['id'])
        if node_key not in node_id_mapping:
            node_id_mapping[node_key] = next_global_id
            next_global_id += 1

        global_data.append({
            "global_node_id": node_id_mapping[node_key],
            "node": row["id"],
            "num_parents": row["num_parents"],
            "k": row["k"],
            "true_split": row["true_split"],
            "found_split": row["found_split"],
            "gmm_bic": row["gmm_bic"],
            "score_diff": row["score_diff"],
            "true_score_diff": row["true_score_diff"],
            "num_iter": row["num_iter"],
            "initial_split": row["initial_split"],
            "per_intv": row["per_intv"],
            "parent_intv": row["parent_intv"],
            "cc_ari": row["cc_ari"], "gmm_ari": row["gmm_ari"], "gmm_ari_res": row["gmm_ari_res"],
            "kmeans_ari": row["kmeans_ari"], "kmeans_ari_res": row["kmeans_ari_res"],
            "spectral_ari": row["spectral_ari"], "spectral_ari_res": row["spectral_ari_res"],
            "cc_nmi": row["cc_nmi"], "gmm_nmi": row["gmm_nmi"], "gmm_nmi_res": row["gmm_nmi_res"],
            "kmeans_nmi": row["kmeans_nmi"], "kmeans_nmi_res": row["kmeans_nmi_res"],
            "spectral_nmi": row["spectral_nmi"], "spectral_nmi_res": row["spectral_nmi_res"],
            "cc_fmi": row["cc_fmi"], "gmm_fmi": row["gmm_fmi"], "gmm_fmi_res": row["gmm_fmi_res"],
            "kmeans_fmi": row["kmeans_fmi"], "kmeans_fmi_res": row["kmeans_fmi_res"],
            "spectral_fmi": row["spectral_fmi"], "spectral_fmi_res": row["spectral_fmi_res"]
        })
print(f" last global id : {next_global_id}")
global_df = pd.DataFrame(global_data)
# --- ARI Score Boxplot (k=2 & true_score_diff <= 0) ---
low_score_df = global_df[
    (global_df["k"] == 2) &
    (pd.to_numeric(global_df["true_score_diff"], errors="coerce") <= 0)
]
low_score = low_score_df[low_score_df["true_split"] == 1]
all_scores = global_df[(global_df["true_split"] == 1) & (global_df["k"] == 2)]
print(f"LOW SCORE CASES: {low_score.shape[0]} . ALL CASES: {all_scores.shape[0]}")

global_df = global_df[(global_df["k"] == 2) & (pd.to_numeric(global_df["true_score_diff"], errors="coerce") > 300)]

# --- Type Conversions ---
numeric_cols = [
    "true_split", "found_split", "cc_ari", "gmm_ari", "kmeans_ari", "spectral_ari",
    "cc_nmi", "gmm_nmi", "kmeans_nmi", "spectral_nmi",
    "cc_fmi", "gmm_fmi", "kmeans_fmi", "spectral_fmi"
]
numeric_cols.extend(["initial_split", "per_intv", "parent_intv"])

global_df[numeric_cols] = global_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# --- ARI Score Boxplot (TP only) ---
filtered_df = global_df[(global_df["true_split"] == 1) & (global_df["found_split"] == 1)]
#ari_cols = ["cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"]
nmi_cols = ["cc_nmi", "gmm_nmi","gmm_nmi_res", "kmeans_nmi",  "kmeans_nmi_res", "spectral_nmi", "spectral_nmi_res"]
#filtered_df[ari_cols] = filtered_df[ari_cols].apply(pd.to_numeric, errors="coerce")
filtered_df[nmi_cols] = filtered_df[nmi_cols].apply(pd.to_numeric, errors="coerce")

plt.figure(figsize=(8, 6))
#sns.boxplot(data=filtered_df[ari_cols])
sns.boxplot(data=filtered_df[nmi_cols])
#plt.title("ARI Score Comparison (True & Found Split)")
plt.title("NMI Score Comparison")
plt.xticks(rotation=15)
#plt.ylabel("ARI Score")
plt.ylabel("NMI Score")
plt.xlabel("Methods")
plt.tight_layout()
plt.show()
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
else:
    print("⚠️ No data to plot for k=2 & true_score_diff ≤ 0.")
# Identify nodes that were already counted in TP or FN
used_nodes = set(global_df[(global_df["true_split"] == 1)]["global_node_id"])
print(f"Nodes with true split (used for TP/FN): {len(used_nodes)}")

# Count TP and FN as usual
TP = global_df[(global_df["true_split"] == 1) & (global_df["found_split"] == 1)].shape[0]
FN = global_df[(global_df["true_split"] == 1) & (global_df["found_split"] == 0)].shape[0]

# Now filter out those nodes from the rest of the DataFrame before computing FP/TN
remaining_df = global_df[~global_df["global_node_id"].isin(used_nodes)]

fp_nodes = []
tn_nodes = []

for node_id, group in remaining_df.groupby("global_node_id"):
    if (group["true_split"] == 0).all():
        if (group["found_split"] == 1).any():
            fp_nodes.append(node_id)
        else:
            tn_nodes.append(node_id)

FP = len(fp_nodes)
TN = len(tn_nodes)

# --- Pie Charts ---
plt.figure(figsize=(6, 6))
plt.pie([FN, TP], labels=["False Negatives", "True Positives"], autopct="%1.1f%%", colors=["red", "green"])
plt.title("FN vs TP")
plt.tight_layout()
plt.show()
#plt.close()

values = [FP, TN]
plt.figure(figsize=(6, 6))
if sum(values) == 0:
    print("Nothing to plot: all values are zero.")
else:
    plt.pie(values, labels=["False Positives", "True Negatives"], autopct="%1.1f%%", colors=["orange", "blue"])
    plt.title("False Positives vs True Negatives")
    plt.show()

# --- Check if Minimum score_diff aligns with correct k ---
correct_k_count = 0
total_k_candidates = 0

for node_id, group in global_df.groupby("global_node_id"):
    if (group["true_split"] == 1).any():
        true_k_vals = group[group["true_split"] == 1]["k"].values
        # Apply conversion logic
        group["score_diff"] = group["score_diff"].apply(
            lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) and x.startswith('[') else float(x)
        )
        preferred_k = group.loc[group["score_diff"].astype(float).idxmin(), "k"]
        total_k_candidates += 1
        if (preferred_k in true_k_vals):
            correct_k_count += 1


print(f"\n✅ Correct k Detection: {correct_k_count}/{total_k_candidates} "
      f"({100 * correct_k_count / total_k_candidates:.2f}%)\n")

'''# --- Line Plot: score_diff vs k for Valid Nodes ---
# Step 1: Get valid node IDs
valid_nodes = global_df.groupby("global_node_id").filter(
    lambda g: (g["true_split"] == 1).any() and (g["found_split"] == 1).any()
)["global_node_id"].unique()

# Step 2: Filter only integer k values
int_k_df = global_df[global_df["k"] == global_df["k"].astype(int)]

# Step 3: Start plotting
plt.figure(figsize=(10, 6))

for node_id in valid_nodes:
    node_df = int_k_df[int_k_df["global_node_id"] == node_id].sort_values("k")

    if not node_df.empty:
        plt.plot(
            node_df["k"],
            node_df["score_diff"],
            marker="o",
            label=f"Node {node_id}"
        )

plt.title("Score Diff vs K for Valid Nodes (Integer k only)")
plt.xlabel("k (int)")
plt.ylabel("score_diff")
plt.legend(title="Node ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()'''

'''# --- ARI Score Boxplot: False Negatives ---
fn_df = global_df[(global_df["true_split"] == 1) & (global_df["found_split"] == 0)]
fn_df[ari_cols] = fn_df[ari_cols].apply(pd.to_numeric, errors="coerce")

plt.figure(figsize=(8, 6))
sns.boxplot(data=fn_df[ari_cols])
plt.title("ARI Score Comparison (False Negatives)")
plt.xticks(rotation=15)
plt.ylabel("ARI Score")
plt.tight_layout()
plt.show()
plt.close()'''

'''# --- Histogram: true_score_diff in FN ---
fn_df["true_score_diff"] = pd.to_numeric(fn_df["true_score_diff"], errors="coerce")
if not fn_df["true_score_diff"].isnull().all():
    plt.figure(figsize=(8, 6))
    sns.histplot(fn_df["true_score_diff"].dropna(), bins=30, kde=True, color="red")
    plt.title("Distribution of true_score_diff (False Negatives)")
    plt.xlabel("true_score_diff")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    plt.close()
else:
    print("⚠️ No valid true_score_diff values found for FN cases.")'''

# --- Final Summary ---
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
