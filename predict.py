import tensorflow as tf
import os
import pandas as pd
import numpy as np
import glob
from data_loader import load_img, standardized_params, load_and_convert_onehot

def predict(config):
    split_folder = config["split_folder"]
    split_idx = config["split_idx"]
    img_folder = config["img_folder"]
    file_col = config["file_col"]
    par_col = config["par_col"]
    tag_col = config["tag_col"]
    tags_to_label = config["tags_to_label"]
    size = config["size"]
    ch = config["ch"]
    par_mean = config["par_mean"]
    par_std = config["par_std"]
    n_classes = config["n_classes"]
    BATCH_SIZE = config["BATCH_SIZE"]
    AUTOTUNE = config["AUTOTUNE"]
    model_folder = config["model_folder"]
    model = config["model"]
    csv_folpath = config["csv_folpath"]

    test_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    test_paths = [os.path.join(img_folder, f) for f in list(test_df[file_col])]
    test_pars = list(test_df[par_col])
    test_labels = [tags_to_label[tag] for tag in list(test_df[tag_col])]
    testset = tf.data.Dataset.from_tensor_slices((test_paths, test_pars, test_labels))
    testset = testset.map(
      lambda path, par, label: ((load_img(path, size, ch), standardized_params(par, par_mean, par_std)), load_and_convert_onehot(label, n_classes))
    ).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    last_model_path = glob.glob(os.path.join(model_folder, "models_{}*".format(split_idx)))[0]
    model.load_weights(last_model_path)
    y_pred = model.predict(testset)
    predict_result = np.argmax(y_pred, axis=1)
    result_arr = np.array([list(test_df["fileName"]), test_pars, test_labels, predict_result]).T
    result_arr = np.concatenate([result_arr, y_pred], axis=1)
    df_result = pd.DataFrame(data=result_arr, columns = ["fileName", "age", "true", "predict", 0, 1])
    df_result.to_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx)), index=False)
    return


