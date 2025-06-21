import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import os

# Enable R-to-Pandas conversion
pandas2ri.activate()

# Load RData file
robjects.r['load']('colon_data.RData')
colon_data = robjects.r['colon_data']


# Function to extract, combine, and save x/y data
def save_experiment(x_key, y_key, folder_name):
    x = list(colon_data.rx2(x_key))
    y = list(colon_data.rx2(y_key))

    # Combine into DataFrame
    df = pd.DataFrame({'x': x, 'y': y})

    # Duplicate the DataFrame (double the rows)
    df = pd.concat([df, df], ignore_index=True)
    # Duplicate again (x4)
    df = pd.concat([df, df], ignore_index=True)
    # Duplicate again (x8)
    df = pd.concat([df, df], ignore_index=True)
    # Duplicate again (x16)
    df = pd.concat([df, df], ignore_index=True)

    # Print row count for verification
    print(f"{folder_name}: Total rows = {len(df)}")

    # Create folder and save as TXT without header
    os.makedirs(folder_name, exist_ok=True)
    output_path = os.path.join(folder_name, 'data1.txt')
    df.to_csv(output_path, sep=',', index=False, header=False)
    print(f"Saved {output_path}")


# Save data for each experiment
save_experiment('x1', 'y1', 'experiment10')
save_experiment('x2', 'y2', 'experiment11')
save_experiment('x3', 'y3', 'experiment12')
