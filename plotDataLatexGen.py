import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# --- Configuration ---
list_folder_names = [
    "test_50_2000_500_0_2_2_3_0_0_0_0",
    "test_50_2000_500_0_2_2_3_0_0_0_1",
    "test_50_2000_500_0_2_2_3_0_0_0_2",
    "test_50_2000_500_0_2_2_3_0_0_4_2",
    "test_50_2000_500_1_2_2_3_0_0_1_0",
    "test_50_2000_500_1_2_2_3_0_0_1_1",
    "test_50_2000_500_1_2_2_3_0_0_2_1",
    "test_50_2000_500_1_2_2_3_0_0_2_2",
    "test_50_2000_500_1_2_2_3_0_0_3_2",
    #"test_50_2000_500_3_2_2_3_0_0_1_0",
    #"test_50_2000_500_3_2_2_3_0_0_1_1",
    #"test_50_2000_500_3_2_2_3_0_0_1_2",
    #"test_50_2000_500_3_2_2_3_0_0_2_1",
    #"test_50_2000_500_3_2_2_3_0_0_2_2",
    #"test_50_2000_500_3_2_2_3_0_0_3_2",

    #"test_50_2000_500_0_1_2_3_0_0_0_0",
    #"test_50_2000_500_0_1_2_3_0_0_0_1",
    #"test_50_2000_500_0_1_2_3_0_0_0_2",
    #"test_50_2000_500_0_1_2_3_0_0_4_2",
    #"test_50_2000_500_1_1_2_3_0_0_1_0",
    #"test_50_2000_500_1_1_2_3_0_0_1_1",
    #"test_50_2000_500_1_1_2_3_0_0_2_1",
    #"test_50_2000_500_1_1_2_3_0_0_2_2",
    #"test_50_2000_500_1_1_2_3_0_0_3_2",
    #"test_50_2000_500_3_1_2_3_0_0_1_0",
    #"test_50_2000_500_3_1_2_3_0_0_1_1",
    #"test_50_2000_500_3_1_2_3_0_0_1_2",
    #"test_50_2000_500_3_1_2_3_0_0_2_1",
    #"test_50_2000_500_3_1_2_3_0_0_2_2",
    #"test_50_2000_500_3_1_2_3_0_0_3_2",
    #"test_50_2000_500_3_1_2_3_0_0_4_2",

    #"test_50_2000_500_0_0_2_3_0_0_0_0",
    #"test_50_2000_500_0_0_2_3_0_0_0_1",
    #"test_50_2000_500_0_0_2_3_0_0_0_2",
    #"test_50_2000_500_1_0_2_3_0_0_0_0",
    #"test_50_2000_500_1_0_2_3_0_0_1_0",
    #"test_50_2000_500_1_0_2_3_0_0_2_0",
    #"test_50_2000_500_1_0_2_3_0_0_2_1",
    #"test_50_2000_500_1_0_2_3_0_0_3_2",
    #"test_50_2000_500_1_0_2_3_0_0_4_2",
    #"test_50_2000_500_3_0_2_3_0_0_0_0",
    #"test_50_2000_500_3_0_2_3_0_0_1_0",
    #"test_50_2000_500_3_0_2_3_0_0_2_1",
    #"test_50_2000_500_3_0_2_3_0_0_3_2"
    #"test_50_200_50_0_2_2_3_0_0_2_1",
    #"test_50_200_50_3_1_2_3_0_0_2_1",
    #"test_50_200_50_3_2_2_3_0_0_2_1",
    #"test_50_400_100_0_2_2_3_0_0_2_1",
    #"test_50_400_100_3_1_2_3_0_0_2_1",
    #"test_50_400_100_3_2_2_3_0_0_2_1",
    #"test_50_2000_500_0_2_2_3_0_0_0_1",
    #"test_50_2000_500_3_1_2_3_0_0_2_1",
    #"test_50_2000_500_3_2_2_3_0_0_2_1",
    #"test_50_4000_1000_0_2_2_3_0_0_2_1",
    #"test_50_4000_1000_3_1_2_3_0_0_2_1",
    #"test_50_4000_1000_3_2_2_3_0_0_2_1"
    ]
base_path = "tests"
base_filename = ""
file_extension = ".csv"
file_indices = range(1, 51)

'''columns_to_keep = [
    "num_parents", "true_split", "found_split", "score_diff", "true_score_diff",
    "per_intv", "parent_intv", "cc_ari", "gmm_ari", "gmm_ari_res",
    "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"
]'''
'''columns_to_keep = [
    "cc_ari", "gmm_ari", "gmm_ari_res", "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"
]'''

columns_to_keep = [
    "cc_ari", "gmm_ari", "kmeans_ari", "spectral_ari"
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
        #df = df[columns_to_keep]
        #df["run_id"] = i
        all_data.append(df)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df[merged_df["num_parents"].isin([1,2])]
    merged_df_pos = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 1)] #(merged_df["true_score_diff"] < 2000)
    merged_df_pos = merged_df_pos[columns_to_keep]
    merged_df_tp_tn = merged_df[(merged_df["true_score_diff"] >= 0) ] #& (merged_df["true_score_diff"] < 2000)
    output_folder = os.path.join(base_path, "final_results") # _low_n
    output_file = os.path.join(output_folder, "scale_periodic_ARI_pos.csv") # _low_n
    merged_df_pos.to_csv(output_file, index=False)
    print(f"Merged results saved as '{output_file}'")
    print(f"ACTUAL TRUE SPLITS : {merged_df[(merged_df["true_split"] == 1)].shape[0]}")
    # TP and FN
    TP = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 1) & (merged_df_tp_tn["found_split"] == 1)].shape[0]
    FN = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 1) & (merged_df_tp_tn["found_split"] == 0)].shape[0]
    print(f"True split {merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 1)].shape[0]}  and TP = {TP}  FN = {FN}")

    # FP and TN
    FP = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 0) & (merged_df_tp_tn["found_split"] == 1)].shape[0]
    TN = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 0) & (merged_df_tp_tn["found_split"] == 0)].shape[0]

    print(f"NOT true split {merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 0)].shape[0]}  and FP = {FP}  TN = {TN}")
    LT40 = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 0) & (merged_df_tp_tn["found_split"] == 1) & (merged_df_tp_tn["per_intv"] < 40.0)].shape[0]
    print(f"FP cases where intv_per is less than 40 :  {LT40}")
    MT40 = merged_df_tp_tn[(merged_df_tp_tn["true_split"] == 0) & (merged_df_tp_tn["found_split"] == 1) & (merged_df_tp_tn["per_intv"] > 40.0)].shape[0]
    print(f"FP cases where intv_per is more than 40 :  {MT40}")
    # Save TP, TN, FP, FN summary to a CSV
    confusion_summary = pd.DataFrame([{
        "True Positives (TP)": TP,
        "False Negatives (FN)": FN,
        "False Positives (FP)": FP,
        "True Negatives (TN)": TN,
        "False Positives LT40": LT40,
        "False Positives MT40": MT40,
    }])

    confusion_file = os.path.join(output_folder, "scale_periodic_confusion_summary.csv") # _low_n
    confusion_summary.to_csv(confusion_file, index=False)
    print(f"Confusion matrix summary saved to '{confusion_file}'")

    # --- ARI Score Boxplot (TP only) ---
    # Convert to long format
    melted_df = pd.melt(
        merged_df_pos,
        #filtered_df,
        #id_vars=["num_parents"],
        # value_vars=nmi_cols,
        value_vars=columns_to_keep,
        var_name="method",
        value_name="score"
    )
    print("Melted df here.")
    pastel_colors = [
        '#7BAFD4',  # blue
        '#66C2A5',  # muted green
        '#FC8D62',  # soft orange
        '#E5C494',  # sand
        '#8DA0CB',  # light blue-violet
        '#E78AC3',  # lavender pink
        '#FFD92F',  # warm yellow
        '#A6D854',  # green-yellow
        '#B3B3B3'  # grey
    ]
    # Minimal styling
    sns.set(style="white", context="notebook", font_scale=1.1)
    mpl.rcParams['axes.linewidth'] = 0.5

    # First plot — True Positives
    plt.figure(figsize=(7.5, 5))
    ax = sns.boxplot(
        data=melted_df,
        x="method",
        y="score",
        #hue="num_parents",
        palette=pastel_colors,
        linewidth=0.6,
        fliersize=0,
        width=0.3,
        boxprops=dict(alpha=0.3),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        medianprops=dict(color='black', linewidth=1.0)
    )

    # Aesthetic cleanup
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=15)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(title="", frameon=False, loc="upper right")
    # Define your custom label mapping
    custom_labels = {
        "cc_ari": "CluC",
        "gmm_ari": "GMM",
        "gmm_ari_res": "GMM (res)",
        "kmeans_ari": "K-means",
        "kmeans_ari_res": "K-means (res)",
        "spectral_ari": "Spectral",
        "spectral_ari_res": "Spectral (res)"
    }

    # Apply custom labels
    ax.set_xticklabels([custom_labels[label.get_text()] for label in ax.get_xticklabels()])
    plt.tight_layout()
    plt.savefig("tests/final_results/scale_periodic_ARI_pos.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # Convert ARI columns to numeric if not already
    fn_df = merged_df[(merged_df["true_split"] == 1) & (merged_df["found_split"] == 0)]
    fn_df[columns_to_keep] = fn_df[columns_to_keep].apply(pd.to_numeric, errors="coerce")
    # low_score_df[nmi_cols] = low_score_df[nmi_cols].apply(pd.to_numeric, errors="coerce")

    # Check if we have any data to plot
    if not fn_df.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=fn_df[columns_to_keep], palette=pastel_colors[:len(columns_to_keep)])
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

