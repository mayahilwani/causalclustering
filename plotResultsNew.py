import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the parent directory containing experiment folders
parent_dir = "./tests/test_50_3000_1000_0_0_2_4_0/"  # Change this to your actual path

# Find all folders matching "experimentX"
experience_folders = sorted(glob.glob(os.path.join(parent_dir, "experiment*")))
# Normalize folder paths
experience_folders = [folder.replace("\\", "/") for folder in experience_folders]

# List to store individual DataFrames
df_list = []

# Read and merge data
for folder in experience_folders:
    file_path = os.path.join(folder, "node_STATS.txt").replace("\\", "/")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)
        df_list.append(df)

# Merge all data
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)

    # Add column names
    merged_df.columns = [
        "id", "num_parents", "k", "true_split", "found_split", "gmm_bic", "score_diff", "true_score_diff", "num_iter",
        "cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res",
        "cc_nmi", "gmm_nmi", "gmm_nmi_res", "kmeans_nmi", "kmeans_nmi_res", "spectral_nmi", "spectral_nmi_res",
        "cc_fmi", "gmm_fmi", "gmm_fmi_res", "kmeans_fmi", "kmeans_fmi_res", "spectral_fmi", "spectral_fmi_res"
    ]

    # Convert columns to numeric where applicable
    numeric_cols = [
        "true_split", "found_split", "cc_ari", "gmm_ari", "kmeans_ari", "spectral_ari",
        "cc_nmi", "gmm_nmi", "kmeans_nmi", "spectral_nmi",
        "cc_fmi", "gmm_fmi", "kmeans_fmi", "spectral_fmi"
    ]
    merged_df[numeric_cols] = merged_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Analysis ---

    # Filter only rows where true_split == 1 and found_split == 1
    filtered_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)]

    res_cols = ["gmm_ari_res", "kmeans_ari_res", "spectral_ari_res"]
    filtered_df[res_cols] = filtered_df[res_cols].apply(pd.to_numeric, errors="coerce")

    # 1. ARI Score Comparison Across Methods (Boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=filtered_df[
        ["cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"]])
    plt.title("ARI Score Comparison Across Methods (only True & Found Splits)")
    plt.ylabel("ARI Score")
    plt.xlabel("Methods")
    plt.xticks(rotation=15)
    plt.show()

    # 2. Pie Plot for FN against TP
    TP = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)].shape[0]
    TN = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 0)].shape[0]
    FP = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 1)].shape[0]
    FN = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)].shape[0]

    plt.figure(figsize=(6, 6))
    plt.pie([FN, TP], labels=["False Negatives (FN)", "True Positives (TP)"], autopct="%1.1f%%",
            colors=["red", "green"])
    plt.title("False Negatives vs. True Positives")
    plt.show()

    # 3. Pie Plot for FP against TN
    plt.figure(figsize=(6, 6))
    plt.pie([FP, TN], labels=["False Positives (FP)", "True Negatives (TN)"], autopct="%1.1f%%",
            colors=["orange", "blue"])
    plt.title("False Positives vs. True Negatives")
    plt.show()

    # Filter for False Negative cases
    fn_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)]

    # Convert relevant columns to numeric
    ari_cols = ["cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"]
    fn_df[ari_cols] = fn_df[ari_cols].apply(pd.to_numeric, errors="coerce")

    # Boxplot for ARI scores in False Negative cases
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=fn_df[ari_cols])
    plt.title("ARI Score Comparison (False Negative Cases)")
    plt.ylabel("ARI Score")
    plt.xlabel("Methods")
    plt.xticks(rotation=15)
    plt.show()

    # Filter for False Negative cases
    fn_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)]

    # Convert 'true_score_diff' to numeric by stripping brackets
    fn_df["true_score_diff"] = fn_df["true_score_diff"].astype(str)  # Convert to string if needed
    fn_df["true_score_diff"] = fn_df["true_score_diff"].str.replace(r"[\[\]]", "", regex=True)  # Remove brackets
    fn_df["true_score_diff"] = pd.to_numeric(fn_df["true_score_diff"], errors="coerce")  # Convert to float
    if not fn_df.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(fn_df["true_score_diff"], bins=30, kde=True, color="red")
        plt.title("Distribution of true_score_diff in False Negative Cases")
        plt.xlabel("True Score Difference")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No valid data for true_score_diff in false negative cases.")

    # Print counts for reference
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")