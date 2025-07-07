import pandas as pd

from outputFilesGen import base_path

# --- CONFIGURATION ---
# List of file paths to merge
file_list = [
    "periodic_2000_500_ARI_pos.csv",
    "periodic_2250_250_ARI_pos.csv",
    # Add more files as needed
]

# Delimiter used in your files (change to ',' if CSV)
delimiter = ","

# Output file path
output_file = "periodic_2.csv"

# --- PROCESSING ---
all_data = []
base_path = "tests/"
for file in file_list:
    try:
        file_path = base_path + file
        df = pd.read_csv(file_path, delimiter=delimiter)
        all_data.append(df)
        print(f"Loaded: {file_path}")
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")

# --- SAVE MERGED FILE ---
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged file saved as: {output_file}")
else:
    print("No valid data loaded.")
