import pandas as pd
import os

# --- Configuration ---
list_folder_names = [
    #"test_50_1500_1000_0_2_2_3_0_0_0_0",
    #"test_50_1500_1000_0_2_2_3_0_0_0_1",
    #"test_50_1500_1000_0_2_2_3_0_0_0_2",
    #"test_50_1500_1000_0_2_2_3_0_0_4_2",
    #"test_50_1500_1000_1_2_2_3_0_0_1_0",
    #"test_50_1500_1000_1_2_2_3_0_0_1_1",
    #"test_50_1500_1000_1_2_2_3_0_0_2_1",
    #"test_50_1500_1000_1_2_2_3_0_0_2_2",
    #"test_50_1500_1000_1_2_2_3_0_0_3_2",
    #"test_50_1500_1000_3_2_2_3_0_0_1_0",
    #"test_50_1500_1000_3_2_2_3_0_0_1_1",
    #"test_50_1500_1000_3_2_2_3_0_0_1_2",
    #"test_50_1500_1000_3_2_2_3_0_0_2_1",
    #"test_50_1500_1000_3_2_2_3_0_0_2_2",
    #"test_50_1500_1000_3_2_2_3_0_0_3_2",
    #"test_50_1500_1000_3_2_2_3_0_0_4_2",

    "test_50_200_50_0_2_2_3_0_0_2_1",
    "test_50_200_50_3_1_2_3_0_0_2_1",
    "test_50_200_50_3_2_2_3_0_0_2_1",
    "test_50_400_100_0_2_2_3_0_0_2_1",
    "test_50_400_100_3_1_2_3_0_0_2_1",
    "test_50_400_100_3_2_2_3_0_0_2_1",
    #"test_50_2000_500_0_2_2_3_0_0_0_1",
    #"test_50_2000_500_3_1_2_3_0_0_2_1",
    #"test_50_2000_500_3_2_2_3_0_0_2_1",
    "test_50_4000_1000_0_2_2_3_0_0_2_1",
    "test_50_4000_1000_3_1_2_3_0_0_2_1",
    "test_50_4000_1000_3_2_2_3_0_0_2_1"
]
base_path = "tests"
base_filename = "experiment"
file_extension = ".txt"
file_indices = range(1, 51)

columns_to_keep = [
    "num_parents", "true_split", "found_split", "score_diff", "true_score_diff",
    "per_intv", "parent_intv", "cc_ari", "gmm_ari", "gmm_ari_res",
    "kmeans_ari", "kmeans_ari_res", "spectral_ari", "spectral_ari_res"
]

row_filter = lambda df: df["k"] == 2  # Example condition

# Loop over each folder
for folder in list_folder_names:
    all_data = []  # New list for each folder
    folder_path = os.path.join(base_path, folder)

    for i in file_indices:
        file_path = os.path.join(folder_path, f"{base_filename}{i}", f"node_STATS{file_extension}")
        print(f"Looking for: {file_path}")
        try:
            df = pd.read_csv(file_path, delimiter=",")
            df = df[row_filter(df)]
            df = df[columns_to_keep]
            df["run_id"] = i
            all_data.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        output_file = os.path.join(folder_path, "merged_selected_results.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Merged results saved as '{output_file}'")
    else:
        print(f"No valid data found for folder: {folder}")
