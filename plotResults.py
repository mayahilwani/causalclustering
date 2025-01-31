import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the stats file
file_path = "C:/Users/ziadh/Documents/CausalGen-Osman/linear/test_2to5_3"
stats_file = f"{file_path}/STATS50.txt"

# Load data into a DataFrame
columns = [
    "id", "intervention", "dag_size", "orig_data", "intv_data",
    "TP", "TN", "FP", "FN", "intv_acc", "avg_cluster_acc", "elapsed_time"
]
df = pd.read_csv(stats_file, header=0, names=columns)

# Step 2: Exclude rows where cluster accuracy is 0
filtered_df = df[df['avg_cluster_acc'] > 0]

# Compute necessary metrics
filtered_df.loc[:, 'precision'] = filtered_df['TP'] / (filtered_df['TP'] + filtered_df['FP'])  # TP / (TP + FP)
filtered_df.loc[:, 'recall'] = filtered_df['TP'] / (filtered_df['TP'] + filtered_df['FN'])     # TP / (TP + FN)
filtered_df.loc[:, 'f1_score'] = 2 * (filtered_df['precision'] * filtered_df['recall']) / (filtered_df['precision'] + filtered_df['recall'])

# Step 3: Generate plots

# 1. Box plot of cluster accuracies
plt.figure(figsize=(8, 5))
sns.boxplot(x=filtered_df['avg_cluster_acc'], color='blue')
plt.title("Box Plot of Cluster Accuracies (Excluding 0)")
plt.xlabel("Cluster Accuracy")
plt.show()

# 2. Box plot of elapsed time
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['elapsed_time'], color='green')
plt.title("Box Plot of Elapsed Time")
plt.xlabel("Elapsed Time (s)")
plt.show()
'''
# 3. Box plot of False Negatives (FN)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['FN'], color='red')
plt.title("Box Plot of False Negatives")
plt.xlabel("Number of False Negatives")
plt.show()

# 4. Box plot of False Positives (FP)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['FP'], color='blue')
plt.title("Box Plot of False Positives")
plt.xlabel("Number of False Positives")
plt.show()

# 5. Box plot of TP-based metrics: Precision, Recall, F1 Score
plt.figure(figsize=(8, 5))
sns.boxplot(data=filtered_df[['precision', 'recall', 'f1_score']], palette="Purples")
plt.title("Box Plot of TP-Based Metrics (Precision, Recall, F1 Score)")
plt.ylabel("Score")
plt.ylim(0, 1)  # Metrics are ratios between 0 and 1
plt.xticks([0, 1, 2], ["Precision", "Recall", "F1 Score"])
plt.show()
'''
