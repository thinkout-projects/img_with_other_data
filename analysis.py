import os
import pandas as pd
from sklearn.metrics import roc_auc_score

def summary(config):
    n_splits = config["n_splits"]
    result_root = config["result_root"]
    csv_folpath = config["csv_folpath"]
    pos_col = config["pos_col"]

    dfs = [pd.read_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx))) for split_idx in range(n_splits)]
    df_summary = pd.concat(dfs)
    df_summary.to_csv(os.path.join(result_root, "miss_summary.csv"), index=False)
    return roc_auc_score(df_summary["true"], df_summary[pos_col])

