import tensorflow as tf
import os
import pandas as pd
import math
from data_loader import load_and_data_augment, standardized_params, load_and_convert_onehot, load_img


def train_regression(config):
    split_folder = config["split_folder"]
    split_idx = config["split_idx"]
    img_folder = config["img_folder"]
    file_col = config["file_col"]
    par_col = config["par_col"]
    tag_col = config["tag_col"]
    tags_to_label = config["tags_to_label"]
    size = config["size"]
    ch = config["ch"]
    basic_process = config["basic_process"]
    extra_process = config["extra_process"]
    par_mean = config["par_mean"] # 非使用
    par_std = config["par_std"] # 非使用
    n_classes = config["n_classes"]
    epochs = config["epochs"]
    BATCH_SIZE = config["BATCH_SIZE"]
    AUTOTUNE = config["AUTOTUNE"]
    optimizer = config["optimizer"]
    model_folder = config["model_folder"]
    model = config["model"]
    loss = config["loss"]
    metrics = config["metrics"]

    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()
    par_mu = train_df[par_col].mean()
    par_sigma = train_df[par_col].std()

    train_paths = [os.path.join(img_folder, f) for f in list(train_df[file_col])]
    train_pars = list(train_df[par_col])
    #train_labels = [tags_to_label[tag] for tag in list(train_df[tag_col])]
    train_tags = list(train_df[tag_col])
    trainset = tf.data.Dataset.from_tensor_slices((train_paths, train_pars, train_tags))
    trainset = trainset.map(
        lambda path, par, tag: ((load_and_data_augment(path, size, ch, basic_process, extra_process),
                                   standardized_params(par, par_mu, par_sigma)),
                                  standardized_params(tag, tag_mu, tag_sigma)),
        num_parallel_calls=AUTOTUNE
    ).cache()
    trainset = trainset.shuffle(buffer_size=len(train_paths)).repeat(epochs).batch(BATCH_SIZE).prefetch(
        buffer_size=AUTOTUNE)

    # valset前処理
    val_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    val_paths = [os.path.join(img_folder, f) for f in list(val_df[file_col])]
    val_pars = list(val_df[par_col])
    #val_labels = [tags_to_label[tag] for tag in list(val_df[tag_col])]
    val_tags = list(val_df[tag_col])
    valset = tf.data.Dataset.from_tensor_slices((val_paths, val_pars, val_tags))
    valset = valset.map(
        lambda path, par, tag: ((load_img(path, size, ch),
                                   standardized_params(par, par_mu, par_sigma)),
                                  standardized_params(tag, tag_mu, tag_sigma)),
        num_parallel_calls=AUTOTUNE
    ).cache()
    valset = valset.repeat(epochs).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    model_path = os.path.join(model_folder, "models_" + str(split_idx) + "_epoch{epoch:02d}.h5")
    mc_cb = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1,
                                               save_weights_only=True, save_best_only=True)
    rl_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1, mode='auto',
                                                 min_delta=0.0001, cooldown=0, min_lr=0)
    es_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto')
    steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    hist = model.fit(trainset, validation_data=valset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                     callbacks=[mc_cb, rl_cb, es_cb])
    return hist


def train_classifier(config):
    split_folder = config["split_folder"]
    split_idx = config["split_idx"]
    img_folder = config["img_folder"]
    file_col = config["file_col"]
    par_col = config["par_col"]
    tag_col = config["tag_col"]
    tags_to_label = config["tags_to_label"]
    size = config["size"]
    ch = config["ch"]
    basic_process = config["basic_process"]
    extra_process = config["extra_process"]
    par_mean = config["par_mean"] # 非使用
    par_std = config["par_std"] # 非使用
    n_classes = config["n_classes"]
    epochs = config["epochs"]
    BATCH_SIZE = config["BATCH_SIZE"]
    AUTOTUNE = config["AUTOTUNE"]
    optimizer = config["optimizer"]
    model_folder = config["model_folder"]
    model = config["model"]
    loss = config["loss"]
    metrics = config["metrics"]

    train_df = pd.read_csv(os.path.join(split_folder, "train_{}.csv".format(split_idx)), encoding="shift-jis")
    tag_mu = train_df[tag_col].mean()
    tag_sigma = train_df[tag_col].std()
    par_mu = train_df[par_col].mean()
    par_sigma = train_df[par_col].std()

    train_paths = [os.path.join(img_folder, f) for f in list(train_df[file_col])]
    train_pars = list(train_df[par_col])
    train_labels = [tags_to_label[tag] for tag in list(train_df[tag_col])]
    #train_tags = list(train_df[tag_col])
    trainset = tf.data.Dataset.from_tensor_slices((train_paths, train_pars, train_labels))
    trainset = trainset.map(
        lambda path, par, label: ((load_and_data_augment(path, size, ch, basic_process, extra_process),
                                   standardized_params(par, par_mu, par_sigma)),
                                  load_and_convert_onehot(label, n_classes)),
        num_parallel_calls=AUTOTUNE
    ).cache()
    trainset = trainset.shuffle(buffer_size=len(train_paths)).repeat(epochs).batch(BATCH_SIZE).prefetch(
        buffer_size=AUTOTUNE)

    # valset前処理
    val_df = pd.read_csv(os.path.join(split_folder, "test_{}.csv".format(split_idx)), encoding="shift-jis")
    val_paths = [os.path.join(img_folder, f) for f in list(val_df[file_col])]
    val_pars = list(val_df[par_col])
    val_labels = [tags_to_label[tag] for tag in list(val_df[tag_col])]
    #val_tags = list(val_df[tag_col])
    valset = tf.data.Dataset.from_tensor_slices((val_paths, val_pars, val_labels))
    valset = valset.map(
        lambda path, par, label: ((load_img(path, size, ch),
                                   standardized_params(par, par_mu, par_sigma)),
                                  load_and_convert_onehot(label, n_classes)),
        num_parallel_calls=AUTOTUNE
    ).cache()
    valset = valset.repeat(epochs).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    model_path = os.path.join(model_folder, "models_" + str(split_idx) + "_epoch{epoch:02d}.h5")
    mc_cb = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1,
                                               save_weights_only=True, save_best_only=True)
    rl_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1, mode='auto',
                                                 min_delta=0.0001, cooldown=0, min_lr=0)
    es_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto')
    steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    hist = model.fit(trainset, validation_data=valset, epochs=epochs, steps_per_epoch=steps_per_epoch,
                     callbacks=[mc_cb, rl_cb, es_cb])
    return hist