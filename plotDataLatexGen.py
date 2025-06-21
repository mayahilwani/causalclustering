import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
'''

    "test_50_2000_500_0_1_2_3_0_0_0_0",
    "test_50_2000_500_0_1_2_3_0_0_0_1",
    #"test_50_2000_500_0_1_2_3_0_0_0_2",
    #"test_50_2000_500_0_1_2_3_0_0_4_2",
    "test_50_2000_500_1_1_2_3_0_0_1_0",
    #"test_50_2000_500_1_1_2_3_0_0_1_1",
    "test_50_2000_500_1_1_2_3_0_0_2_1",
    #"test_50_2000_500_1_1_2_3_0_0_2_2",
    "test_50_2000_500_1_1_2_3_0_0_3_2",
    "test_50_2000_500_3_1_2_3_0_0_1_0",
    #"test_50_2000_500_3_1_2_3_0_0_1_1",
    "test_50_2000_500_3_1_2_3_0_0_1_2",
    "test_50_2000_500_3_1_2_3_0_0_2_1",
    #"test_50_2000_500_3_1_2_3_0_0_2_2",
    "test_50_2000_500_3_1_2_3_0_0_3_2",
    "test_50_2000_500_3_1_2_3_0_0_4_2",
'''
list_folder_names = [
    "test_50_2000_500_0_0_2_3_0_0_0_0",
    "test_50_2000_500_0_0_2_3_0_0_0_1",
    "test_50_2000_500_0_0_2_3_0_0_0_2",
    "test_50_2000_500_1_0_2_3_0_0_0_0",
    "test_50_2000_500_1_0_2_3_0_0_1_0",
    "test_50_2000_500_1_0_2_3_0_0_2_0",
    "test_50_2000_500_1_0_2_3_0_0_2_1",
    "test_50_2000_500_1_0_2_3_0_0_3_2",
    "test_50_2000_500_1_0_2_3_0_0_4_2",
    "test_50_2000_500_3_0_2_3_0_0_0_0",
    "test_50_2000_500_3_0_2_3_0_0_1_0",
    "test_50_2000_500_3_0_2_3_0_0_2_1",
    "test_50_2000_500_3_0_2_3_0_0_3_2"
]
base_path = "tests"
base_filename = ""
file_extension = ".csv"
file_indices = range(1, 51)

columns_to_keep = [
    "num_parents", "true_split", "found_split", "score_diff", "true_score_diff",
    "per_intv", "parent_intv", "cc_ari", "gmm_ari", "gmm_ari_res",
    "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"
]

row_filter = lambda df: df["true_score_diff"] > 0  # Example condition
all_data = []
# Loop over each folder
for folder in list_folder_names:
    folder_path = os.path.join(base_path, folder)
    file_path = os.path.join(folder_path, f"merged_selected_results{file_extension}")
    #print(f"Looking for: {file_path}")
    try:
        df = pd.read_csv(file_path, delimiter=",")
        #df = df[row_filter(df)]
        df = df[columns_to_keep]
        #df["run_id"] = i
        all_data.append(df)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df_pos = merged_df[merged_df["true_score_diff"] > 0]
    output_file = os.path.join(base_path, "linear_2000_500_ARI_pos.csv") # _low_n
    merged_df_pos.to_csv(output_file, index=False)
    print(f"Merged results saved as '{output_file}'")
    TP = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)].shape[0]
    FN = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)].shape[0]
    print(f"True split {merged_df[(merged_df["true_split"] == 1)].shape[0]}  and TP = {TP}  FN = {FN}")

    # FP and TN
    FP = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 1)].shape[0]
    TN = merged_df[(merged_df["true_split"] == 0) & (merged_df["found_split"] == 0)].shape[0]
    print(f"NOT true split {merged_df[(merged_df["true_split"] == 0)].shape[0]}  and FP = {FP}  TN = {TN}")
    # Save TP, TN, FP, FN summary to a CSV
    confusion_summary = pd.DataFrame([{
        "True Positives (TP)": TP,
        "False Negatives (FN)": FN,
        "False Positives (FP)": FP,
        "True Negatives (TN)": TN
    }])

    confusion_file = os.path.join(base_path, "confusion_summary_linear_2000_500.csv") # _low_n
    confusion_summary.to_csv(confusion_file, index=False)
    print(f"Confusion matrix summary saved to '{confusion_file}'")

    # --- ARI Score Boxplot (TP only) ---
    filtered_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)]
    ari_cols = ["cc_ari", "gmm_ari",  "kmeans_ari", "spectral_ari"] # "gmm_ari_res", "kmeans_ari_res", , "spectral_ari_res"
    # nmi_cols = ["cc_nmi", "gmm_nmi","gmm_nmi_res", "kmeans_nmi",  "kmeans_nmi_res", "spectral_nmi", "spectral_nmi_res"]

    '''tp_1 = tp_df[tp_df["num_parents"] == 1]
    print("Non-NaN values for num_parents = 1:")
    print(tp_1[nmi_cols].notna().sum())'''

    # Convert to long format
    melted_df = pd.melt(
        merged_df_pos,
        #filtered_df,
        id_vars=["num_parents"],
        # value_vars=nmi_cols,
        value_vars=ari_cols,
        var_name="method",
        value_name="score"
    )
    print("Melted df here.")

    plt.figure(figsize=(8, 6))
    # sns.boxplot(data=filtered_df[ari_cols])
    sns.boxplot(data=melted_df,
                x="method",
                y="score",
                hue="num_parents")
    plt.title("ARI Score Comparison (True & Found Split)")
    # plt.title("NMI Score Comparison for TP")
    plt.xticks(rotation=15)
    plt.ylabel("ARI Score")
    # plt.ylabel("NMI Score")
    plt.xlabel("Methods")
    plt.tight_layout()
    plt.show()
    print('First box plot with all TP cases')

    # Convert ARI columns to numeric if not already
    fn_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)]
    fn_df[ari_cols] = fn_df[ari_cols].apply(pd.to_numeric, errors="coerce")
    # low_score_df[nmi_cols] = low_score_df[nmi_cols].apply(pd.to_numeric, errors="coerce")

    # Check if we have any data to plot
    if not fn_df.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=fn_df[ari_cols])
        # sns.boxplot(data=low_score_df[nmi_cols])
        # plt.title("ARI Score Comparison (k=2 & true_score_diff ≤ 0)")
        plt.title("NMI Score Comparison (k=2 & true_score_diff ≤ 0)")
        plt.xticks(rotation=15)
        plt.ylabel("ARI Score")
        # plt.ylabel("NMI Score")
        plt.xlabel("Methods")
        plt.tight_layout()
        plt.show()
        print('Second box plot with negative true score cases')

else:
    print(f"No valid data found for folder: {folder}")

# For the first filter I want ARI score comparision for the different methods for cases of TP and a TP TN FP FN display.
# For the second filter I want  ARI score comparision for the different methods for cases of TP and a TP TN FP FN display.

