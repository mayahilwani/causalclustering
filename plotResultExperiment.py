import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Path to the parent directory containing experiment folders
parent_dir = "./inheritancetests/test_30_2000_1000_0_1_2_2_0"  # Change this as needed

# Get all experiment folders
experience_folders = sorted(glob.glob(os.path.join(parent_dir, "experiment*")))
experience_folders = [folder.replace("\\", "/") for folder in experience_folders]

df_list = []

# Read and combine all node_STATS.txt files
for folder in experience_folders:
    file_path = os.path.join(folder, "node_STATS.txt").replace("\\", "/")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=0)
        df_list.append(df)

if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)

    # Add column names
    merged_df.columns = [
        "id", "num_parents", "k", "true_split", "found_split", "gmm_bic", "score_diff", "true_score_diff", "num_iter",
        "cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res",
        "cc_nmi", "gmm_nmi", "gmm_nmi_res", "kmeans_nmi", "kmeans_nmi_res", "spectral_nmi", "spectral_nmi_res",
        "cc_fmi", "gmm_fmi", "gmm_fmi_res", "kmeans_fmi", "kmeans_fmi_res", "spectral_fmi", "spectral_fmi_res"
    ]

    # Clean bracketed float values if needed
    def extract_first_float(val):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(val))
        return float(match.group()) if match else np.nan

    for col in ["gmm_bic", "score_diff", "true_score_diff"]:
        merged_df[col] = merged_df[col].apply(extract_first_float)

    # Clean up nested values (like "[3647.57]") if they appear
    merged_df["gmm_bic"] = merged_df["gmm_bic"].astype(str).str.extract(r"([\d\.\-eE]+)").astype(float)
    merged_df["score_diff"] = merged_df["score_diff"].astype(str).str.extract(r"([\d\.\-eE]+)").astype(float)
    merged_df["true_score_diff"] = merged_df["true_score_diff"].astype(str).str.extract(r"([\d\.\-eE]+)").astype(float)

    # Convert key fields to numeric
    merged_df[["id", "true_split", "found_split"]] = merged_df[["id", "true_split", "found_split"]].apply(pd.to_numeric)

    # Count TP and FN
    tp_count = ((merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)).sum()
    fn_count = ((merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)).sum()

    print(f"True Positives (TP): {tp_count}")
    print(f"False Negatives (FN): {fn_count}")

    # Pie Plot of TP vs FN
    plt.figure(figsize=(6, 6))
    plt.pie([tp_count, fn_count],
            labels=["TP", "FN"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#a1dab4", "#41b6c4"])
    plt.title("True Positives vs False Negatives")
    plt.axis("equal")  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.show()

    # Percentage of found_split == 1 for node id == 2
    node_2_df = merged_df[merged_df["id"] == 2]
    if not node_2_df.empty:
        percent_found_1_node2 = (node_2_df["found_split"] == 1).mean() * 100
        print(f"Percentage of found_split == 1 for node 2: {percent_found_1_node2:.2f}%")
        counts = node_2_df["found_split"].value_counts().reindex([1, 0], fill_value=0)
        labels = ["found_split = 1", "found_split = 0"]
        colors = ["#66c2a5", "#fc8d62"]

        plt.figure(figsize=(6, 6))
        plt.pie(counts,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors)
        plt.title("found_split distribution for node id = 2")
        plt.axis("equal")
        plt.show()
    else:
        print("No data found for node with id = 2")

    # Box Plot of cc_nmi for node 2 where found_split == 1
    node2_found_df = node_2_df[node_2_df["found_split"] == 1]

    if not node2_found_df.empty:
        node2_found_df.loc[:, "cc_nmi"] = pd.to_numeric(node2_found_df["cc_nmi"], errors="coerce")

        plt.figure(figsize=(5, 6))
        sns.boxplot(y=node2_found_df["cc_nmi"], color="#8da0cb")
        plt.title("cc_nmi for Node 2 (found_split == 1)")
        plt.ylabel("cc_nmi Score")
        plt.tight_layout()
        plt.show()
    else:
        print("No entries for node id = 2 with found_split == 1")

    # Box Plot of cc_nmi for true_split == 1 and found_split == 1
    filtered_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)]

    if not filtered_df.empty:
        filtered_df.loc[:, "cc_nmi"] = pd.to_numeric(filtered_df["cc_nmi"], errors="coerce")

        plt.figure(figsize=(5, 6))
        sns.boxplot(y=filtered_df["cc_nmi"], color="#66c2a5")
        plt.title("cc_nmi for true_split == 1 and found_split == 1")
        plt.ylabel("cc_nmi Score")
        plt.tight_layout()
        plt.show()
    else:
        print("No entries for true_split == 1 and found_split == 1")
