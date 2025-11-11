import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

# # --- seqFISH+  ---
pred = pd.read_csv("./output/predict_result.csv", index_col=0)
true = pd.read_csv("./data/ST_data/ST_ground_truth.tsv", sep='\t', index_col=0) 

# --- seqFISH  ---
# pred = pd.read_csv("./output/predict_result.csv", index_col=0)
# true = pd.read_csv("./seqfish/ST_data/ST_ground_truth.tsv", sep='\t', index_col=0) 

# # --- merFISH  ---
# pred = pd.read_csv("./output/predict_result.csv", index_col=0)
# true = pd.read_csv("./merfish/ST_data/ST_ground_truth.tsv", sep='\t', index_col=0) 

common_cols = pred.columns.intersection(true.columns)
pred = pred.loc[true.index, common_cols]
true = true.loc[true.index, common_cols]


P = pred.values + 1e-10  
Q = true.values + 1e-10

# --- Jensen-Shannon Divergence (per spot) ---
def jsd(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(rel_entr(P, M)) + 0.5 * np.sum(rel_entr(Q, M))

jsd_scores = [jsd(P[i], Q[i]) for i in range(P.shape[0])]
avg_jsd = np.mean(jsd_scores)

# --- Root Mean Squared Error ---
rmse = np.sqrt(mean_squared_error(Q, P))

# --- Spearman's Rank Correlation (per spot) ---
spearman_scores = []
for i in range(P.shape[0]):
    r, _ = spearmanr(P[i], Q[i])
    spearman_scores.append(r)
avg_spearman = np.nanmean(spearman_scores)

# --- Print Results ---
print(f"Average JSD:           {avg_jsd:.4f}")
print(f"RMSE:                  {rmse:.4f}")
print(f"Average Spearman R:    {avg_spearman:.4f}")







