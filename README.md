# img_with_other_data
画像と他データを一緒に入れた場合の解析を行います



## 使い方

main.pyのあるディレクトリに画像の入ったフォルダとcsvファイルを入れる

→main.pyを編集
画像フォルダ名（img_folder）・csvファイル名（data_csv）・画像/他データ/目的変数/IDのカラム（file_col, par_col, tag_col, ID_col）・分類数（n_classes:1なら回帰、2以上なら分類）を指定

→main.pyを実行

→result_summary.csvのvalueを見る（分類ならAUC、回帰なら相関係数が出力される）



## 構成ファイル

analysis.py：AUC/相関係数の計算

data_augment.py：画像の前処理

data_loader.py：@tf.fuctionを伴うデータの読み込み、標準化処理

k_fold_split.py：層化k分割

learning.py：学習

main.py：メイン処理

models.py：モデル構成（VGG16）

predict.py：予測

utils.py：ツール