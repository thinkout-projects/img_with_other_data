import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

def summary_classification(config):
    n_splits = config["n_splits"]
    result_root = config["result_root"]
    csv_folpath = config["csv_folpath"]
    pos_col = config["pos_col"]

    dfs = [pd.read_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx))) for split_idx in range(n_splits)]
    df_summary = pd.concat(dfs)
    df_summary.to_csv(os.path.join(result_root, "miss_summary.csv"), index=False)
    return roc_auc_score(df_summary["true"], df_summary[pos_col])

def summary_regression(config):
    n_splits = config["n_splits"]
    result_root = config["result_root"]
    csv_folpath = config["csv_folpath"]
    #pos_col = config["pos_col"]

    dfs = [pd.read_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx))) for split_idx in range(n_splits)]
    df_summary = pd.concat(dfs)
    df_summary.to_csv(os.path.join(result_root, "miss_summary.csv"), index=False)
    return stats.pearsonr(df_summary["true"], df_summary["predict"])[0]

# 引っ張ってきただけ
def summary_analysis_regression(miss_summary_file, summary_file, fig_file):
    df = pd.read_csv(miss_summary_file, encoding="utf-8")
    y_true = df["true"]
    y_pred = df["predict"]
    r, p = stats.pearsonr(y_true, y_pred)
    print("相関係数")
    print(r, p)
    df_out = pd.DataFrame()
    df_out["pearsonr"] = r, p
    df_out.to_csv(summary_file, index=False, encoding="utf-8")

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('Ground value')
    plt.ylabel('Predict Value')
    plt.title('Prediction')
    plt.savefig(fig_file)
    plt.close()