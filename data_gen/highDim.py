from pandas.plotting import parallel_coordinates
import pandas as pd
from sklearn.datasets import load_iris

# Load a high-dimensional dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Create a parallel coordinates plot
parallel_coordinates(df, 'species')