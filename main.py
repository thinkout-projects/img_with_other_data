import os
import pandas as pd
import tensorflow as tf
import optuna
from utils import file_check, plot_history, model_delete, printWithDate
from k_fold_split import Stratified_group_k_fold
from data_augment import ExtraProcess, BasicProcess
from models import concat_model
from learning import train
from predict import predict
from analysis import summary




def main():
    printWithDate("main() function is started")
    img_folder = "Axial_img"
    split_folder = "split"
    data_csv = "data_ensui.csv"
    file_col = "fileName"
    ID_col = "None"
    hasID = False if ID_col == "None" else True
    par_col = "age"
    tag_col = "CXL"
    tags_to_label = {0: 0, 1: 1}
    n_classes = 2
    n_splits = 5
    size = [224, 224]
    ch = 3
    basic_process = BasicProcess()
    extra_process = ExtraProcess()
    n_trials = 10
    epochs = 40
    print(str(n_splits), "-fold", str(n_trials), "-trials", str(epochs) , "epochs")
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    loss = "binary_crossentropy"
    metrics = "accuracy"
    result_roothead = "result"
    model_folder = "model"
    figure_folder = "figure"
    csv_folder = "csv"
    pos_col = "1"
    result_summary_csv = "result_summary.csv"

    df = pd.read_csv(data_csv, encoding="shift-jis")
    assert file_check(list(df[file_col]), os.listdir(img_folder))

    par_mean = df[par_col].mean()
    par_std = df[par_col].std()

    printWithDate("spliting dataset")
    sgkf = Stratified_group_k_fold(
        csv_config={"file_col": file_col, "tag_col": tag_col, "ID_col": ID_col},
        n_splits=n_splits,
        shuffle=True,
        split_info_folder = split_folder
    )
    df_train_list, df_test_list = sgkf.k_fold_classifier(df)

    def each_cycle(trial):
        cycle_idx = trial.number
        lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        hidden_dig = trial.suggest_int('hidden_dig', 6, 8)
        img_ratio = trial.suggest_uniform('img_ratio', 0.2, 0.8)

        printWithDate("{}cycle start".format(cycle_idx))
        result_root = result_roothead + "_" + str(cycle_idx).zfill(3)
        model_folpath= os.path.join(result_root, model_folder)
        figure_folpath = os.path.join(result_root, figure_folder)
        csv_folpath = os.path.join(result_root, csv_folder)
        os.makedirs(model_folpath, exist_ok=True)
        os.makedirs(figure_folpath, exist_ok=True)
        os.makedirs(csv_folpath, exist_ok=True)
        optimizer = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        for split_idx in range(n_splits):
            printWithDate(str(cycle_idx) + "cycle, " + str(split_idx) + "split training start")
            tf.keras.backend.clear_session()
            model = concat_model((224, 224, 3), (1), int(2**hidden_dig), img_ratio, 2)
            # 訓練
            train_config = {
                "split_folder": split_folder,
                "split_idx": split_idx,
                "img_folder": img_folder,
                "file_col": file_col,
                "par_col": par_col,
                "tag_col": tag_col,
                "tags_to_label": tags_to_label,
                "size": size,
                "ch": ch,
                "basic_process": basic_process,
                "extra_process": extra_process,
                "par_mean": par_mean,
                "par_std": par_std,
                "n_classes": n_classes,
                "epochs": epochs,
                "BATCH_SIZE": BATCH_SIZE,
                "AUTOTUNE": AUTOTUNE,
                "optimizer": optimizer,
                "model_folder": model_folpath,
                "model": model,
                "loss": loss,
                "metrics": metrics
            }
            history = train(train_config)
            printWithDate(str(cycle_idx) + "cycle, " + str(split_idx) + "split training finish")
            # plotおよび削除
            plot_fpath = os.path.join(figure_folpath, "history_{}.png".format(split_idx))
            plot_history(history, plot_fpath)
            model_delete(model_folpath, split_idx)
            printWithDate(str(cycle_idx) + "cycle, " + str(split_idx) + "split evaluation start")
            predict_config = {
                "split_folder": split_folder,
                "split_idx": split_idx,
                "img_folder": img_folder,
                "file_col": file_col,
                "par_col": par_col,
                "tag_col": tag_col,
                "tags_to_label": tags_to_label,
                "size": size,
                "ch": ch,
                "par_mean": par_mean,
                "par_std": par_std,
                "n_classes": n_classes,
                "BATCH_SIZE": BATCH_SIZE,
                "AUTOTUNE": AUTOTUNE,
                "model_folder": model_folpath,
                "model": model,
                "csv_folpath": csv_folpath
            }
            predict(predict_config)
        printWithDate(str(cycle_idx) + "cycle, summary start")
        summary_config = {
            "n_splits": n_splits,
            "result_root": result_root,
            "csv_folpath": csv_folpath,
            "pos_col": pos_col
        }
        return summary(summary_config)

    study = optuna.create_study()
    study.optimize(each_cycle, n_trials=n_trials)
    df_study = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df_study.to_csv(result_summary_csv, index=False)


if __name__ == "__main__":
    main()
