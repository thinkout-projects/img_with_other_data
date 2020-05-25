import sys
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pytz import timezone
from datetime import datetime


def file_check(fs0, fs1):
    for f in fs0:
        if f not in fs1:
            return False
    return True


def plot_history_classifier(history, fpath):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    sns.set_style("ticks")
    f, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.lineplot(x=list(range(1, len(acc) + 1)), y=acc, label="acc", ax=axs[0])
    sns.lineplot(x=list(range(1, len(val_acc) + 1)), y=val_acc, label="val_acc", ax=axs[0])
    axs[0].set(xlabel="epochs", ylabel="accuracy", title="Training and Validation acc")
    sns.lineplot(x=list(range(1, len(loss) + 1)), y=loss, label="loss", ax=axs[1])
    sns.lineplot(x=list(range(1, len(val_loss) + 1)), y=val_loss, label="val_loss", ax=axs[1])
    axs[1].set(xlabel="epochs", ylabel="loss", title="Training and Validation loss")
    plt.savefig(fpath)
    plt.close()
    return


def plot_history_regression(history, fpath):
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    sns.set_style("ticks")
    f, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.lineplot(x=list(range(1, len(mae) + 1)), y=mae, label="mae", ax=axs[0])
    sns.lineplot(x=list(range(1, len(val_mae) + 1)), y=val_mae, label="val_mae", ax=axs[0])
    axs[0].set(xlabel="epochs", ylabel="MAE", title="Training and Validation MAE")
    sns.lineplot(x=list(range(1, len(loss) + 1)), y=loss, label="loss", ax=axs[1])
    sns.lineplot(x=list(range(1, len(val_loss) + 1)), y=val_loss, label="val_loss", ax=axs[1])
    axs[1].set(xlabel="epochs", ylabel="loss", title="Training and Validation loss")
    plt.savefig(fpath)
    plt.close()
    return


def model_delete(model_folder, split_idx):
    model_paths = glob.glob(os.path.join(model_folder, "models_{}*".format(split_idx)))
    for model_path in model_paths[:-1]:
      os.remove(model_path)
    return


def printWithDate(*printee):
    '''
    通常のprint文に加え、[ yyyy/mm/dd hh:mm:ss ] を文頭に挿入する。
    ただし、`sep`, `end`, `file`, `flush`は使用不可能。
    '''
    print("[", datetime.now().astimezone(timezone('Asia/Tokyo'))
          .strftime("%Y/%m/%d %H:%M:%S"), "] ", end="")
    for i in printee:
        print(i, end="")
    print()
    return


def result_delete_check():
    delete_flag = False
    current_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    result_directory = 'result_000'
    if os.path.exists(os.path.join(current_directory, result_directory)):
        delete_ans = input("過去のresultがフォルダ内に残っています。このまま削除して続行しますか？(yなら続行)")
        if delete_ans == 'y' or delete_ans == 'Y':
            for i in range(1000):
                result_directory = 'result_' + str(i).zfill(3)
                if os.path.exists(os.path.join(current_directory, result_directory)):
                    shutil.rmtree(result_directory)
                else:
                    break
        else:
            exit()