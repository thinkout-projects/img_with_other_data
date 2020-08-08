import tensorflow as tf
import os
import pandas as pd
import numpy as np
import glob
from data_loader import load_img, standardized_params, load_and_convert_onehot

def predict_regression(config):
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

    # 標準化predictの逆変換用
    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()
    par_mu = train_df[par_col].mean()
    par_sigma = train_df[par_col].std()

    test_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    test_paths = [os.path.join(img_folder, f) for f in list(test_df[file_col])]
    test_pars = list(test_df[par_col])
    test_tags = list(test_df[tag_col])
    testset = tf.data.Dataset.from_tensor_slices((test_paths, test_pars, test_tags))
    testset = testset.map(
      lambda path, par, tag: ((load_img(path, size, ch), standardized_params(par, par_mu, par_sigma)),
                                  standardized_params(tag, tag_mu, tag_sigma))
    ).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    last_model_path = glob.glob(os.path.join(model_folder, "models_{}*".format(split_idx)))[0]
    model.load_weights(last_model_path)
    predict_result = model.predict(testset)
    predict_result = [(i[0]*tag_sigma + tag_mu) for i in predict_result]
    result_arr = np.array([list(test_df[file_col]), test_pars, test_tags, predict_result]).T
    df_result = pd.DataFrame(data=result_arr, columns = [file_col, par_col, "true", "predict"])
    df_result.to_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx)), index=False)
    return


def non_par_predict_regression(config):
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

    # 標準化predictの逆変換用
    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()

    test_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    test_paths = [os.path.join(img_folder, f) for f in list(test_df[file_col])]
    test_tags = list(test_df[tag_col])
    testset = tf.data.Dataset.from_tensor_slices((test_paths, test_tags))
    testset = testset.map(
      lambda path, tag: ((load_img(path, size, ch)),
                            standardized_params(tag, tag_mu, tag_sigma))
    ).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



    last_model_path = glob.glob(os.path.join(model_folder, "models_{}*".format(split_idx)))[0]
    model.load_weights(last_model_path)
    predict_result = model.predict(testset)
    predict_result = [(i[0]*tag_sigma + tag_mu) for i in predict_result]
    result_arr = np.array([list(test_df[file_col]), test_tags, predict_result]).T
    df_result = pd.DataFrame(data=result_arr, columns = [file_col, "true", "predict"])
    df_result.to_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx)), index=False)
    return




def predict_classification(config):
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

    # 標準化predictの逆変換用
    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()
    par_mu = train_df[par_col].mean()
    par_sigma = train_df[par_col].std()

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
    result_arr = np.array([list(test_df[file_col]), test_pars, test_labels, predict_result]).T
    result_arr = np.concatenate([result_arr, y_pred], axis=1)
    df_result = pd.DataFrame(data=result_arr, columns = [file_col, par_col, "true", "predict", 0, 1])
    df_result.to_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx)), index=False)
    return


def non_par_predict_classification(config):
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

    # 標準化predictの逆変換用
    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()

    test_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    test_paths = [os.path.join(img_folder, f) for f in list(test_df[file_col])]
    test_labels = [tags_to_label[tag] for tag in list(test_df[tag_col])]
    testset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    testset = testset.map(
      lambda path, label: ((load_img(path, size, ch)), load_and_convert_onehot(label, n_classes))
    ).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    last_model_path = glob.glob(os.path.join(model_folder, "models_{}*".format(split_idx)))[0]
    model.load_weights(last_model_path)
    y_pred = model.predict(testset)
    predict_result = np.argmax(y_pred, axis=1)
    result_arr = np.array([list(test_df[file_col]), test_labels, predict_result]).T
    result_arr = np.concatenate([result_arr, y_pred], axis=1)
    df_result = pd.DataFrame(data=result_arr, columns = [file_col, "true", "predict", 0, 1])
    df_result.to_csv(os.path.join(csv_folpath, "miss_{}.csv".format(split_idx)), index=False)
    return