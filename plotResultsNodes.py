import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
# 12.6 km
# Path to the parent directory containing experience folders
parent_dir = "C:/Users/ziadh/Documents/CausalGen-Osman/periodic/test_2to5_3"  # Change this to your actual path

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
        "id", "num_parents", "true_split", "found_split", "score_diff",
        "num_iter", "method_acc", "gmm_acc", "kmeans_acc"
    ]
    print("Columns in DataFrame:", merged_df.columns)
    # Convert 'true_split' and 'found_split' to integers
    merged_df["true_split"] = pd.to_numeric(merged_df["true_split"], errors="coerce")
    merged_df["found_split"] = pd.to_numeric(merged_df["found_split"], errors="coerce")
    # Convert accuracy columns to floats
    accuracy_cols = ["method_acc", "gmm_acc", "kmeans_acc"]
    for col in accuracy_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
    #print(merged_df.shape)

    # Save to CSV (optional)
    #merged_df.to_csv("merged_node_stats.csv", index=False)

    # --- Analysis ---

    ## 1. Percentage of true interventions found
    total_true_interventions = merged_df[merged_df["true_split"] == 1].shape[0]
    print('TOTAL TRUE ', total_true_interventions)
    found_true_interventions = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)].shape[0]
    print('FOUND TRUE ', found_true_interventions)

    if total_true_interventions > 0:
        found_percentage = (found_true_interventions / total_true_interventions) * 100

        plt.figure(figsize=(6, 6))
        plt.pie([found_true_interventions, total_true_interventions - found_true_interventions],
                labels=["Found", "Not Found"], autopct="%1.1f%%", colors=["green", "red"])
        plt.title("Percentage of True Interventions Found")
        plt.show()

    ## 2. Compare method accuracy for found interventions where true_split is 1
    found_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)]

    if not found_df.empty:
        # Plot accuracy comparison
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=found_df[["method_acc", "gmm_acc", "kmeans_acc"]])
        plt.title("Accuracy Comparison for True and Found Interventions")
        plt.ylabel("Accuracy")
        plt.xlabel("Methods")
        plt.show()

        print(f"Percentage of true interventions found: {found_percentage:.2f}%")
    else:
        print("No data found!")
