import pandas as pd

df = pd.read_csv("linear_ALL_ARI_pos.csv")
ari_cols = ["cc_ari", "gmm_ari", "kmeans_ari", "spectral_ari"]  #

summary = []

for col in ari_cols:
    values = df[col].dropna()
    '''summary.append([
        values.quantile(0.05),  # lower whisker
        values.quantile(0.25),  # lower quartile
        values.quantile(0.50),  # median
        values.quantile(0.75),  # upper quartile
        values.quantile(0.95)   # upper whisker
    ])'''
    #summary_df = pd.DataFrame(summary)
    values.to_csv(f"linear_all_{col}_ari_latex.tsv", index=False, header=False, sep="\t")

