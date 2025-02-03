import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the parent directory containing experience folders
parent_dir = "C:/Users/ziadh/Documents/CausalGen-Osman/norm/non_linear_2/test_3to1_1"  # Change this to your actual path

# Find all folders matching "experienceX"
experience_folders = sorted(glob.glob(os.path.join(parent_dir, "expirement*")))
# Normalize folder paths
experience_folders = [folder.replace("\\", "/") for folder in experience_folders]
#print(experience_folders)
# List to store individual DataFrames
df_list = []

# Read and merge data
for folder in experience_folders:
    file_path = os.path.join(folder, "node_STATS.txt")
    file_path = file_path.replace("\\", "/")
    if os.path.exists(file_path):
        #print('path exists: ', file_path)
        df = pd.read_csv(file_path, header=None)
        df_list.append(df)
    #else: print('path does not exist')

# Merge all data
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)

    # Add column names
    merged_df.columns = [
        "id", "num_parents", "true_split", "found_split", "gmm_bic", "score_diff", "true_score_diff", "num_iter", "method_acc", "gmm_acc", "gmm_acc_res", "kmeans_acc", "kmeans_acc_res", "f1", "gmm_f1", "gmm_f1_res", "kmeans_f1", "kmeans_f1_res"
    ]
    print("Columns in DataFrame:", merged_df.columns)
    # Convert 'true_split' and 'found_split' to integers
    merged_df["true_split"] = pd.to_numeric(merged_df["true_split"], errors="coerce")
    merged_df["found_split"] = pd.to_numeric(merged_df["found_split"], errors="coerce")
    # Convert columns to floats
    numeric_cols = ["true_split", "found_split", "method_acc", "gmm_acc", "kmeans_acc",
                    "f1", "gmm_f1", "gmm_f1_res", "kmeans_f1", "kmeans_f1_res"]
    merged_df[numeric_cols] = merged_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    #print(merged_df.shape)

    # Save to CSV (optional)
    #merged_df.to_csv("merged_node_stats.csv", index=False)

    # --- Analysis ---

    # Filter only rows where true_split == 1 and found_split == 1
    filtered_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)]

    # 1. F1 Score Comparison Across Methods (Boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=filtered_df[["f1", "gmm_f1", "gmm_f1_res", "kmeans_f1", "kmeans_f1_res"]])
    plt.title("F1 Score Comparison Across Methods (True & Found Splits)")
    plt.ylabel("F1 Score")
    plt.xlabel("Methods")
    plt.xticks(rotation=15)
    plt.show()

    # 2. TP, TN, FP, FN Calculation
    TP = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)].shape[0]
    TN = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 0)].shape[0]
    FP = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 1)].shape[0]
    FN = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)].shape[0]

    # 3. Pie Plot for FN against TP
    plt.figure(figsize=(6, 6))
    plt.pie([FN, TP], labels=["False Negatives (FN)", "True Positives (TP)"], autopct="%1.1f%%",
            colors=["red", "green"])
    plt.title("False Negatives vs. True Positives")
    plt.show()

    # 4. Pie Plot for FP against TN
    plt.figure(figsize=(6, 6))
    plt.pie([FP, TN], labels=["False Positives (FP)", "True Negatives (TN)"], autopct="%1.1f%%",
            colors=["orange", "blue"])
    plt.title("False Positives vs. True Negatives")
    plt.show()

    # Print counts for reference
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
