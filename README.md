# img_with_other_data
画像と他データを一緒に入れた場合の解析を行います

現在、コーディング中

コーディング先、URL

https://drive.google.com/drive/u/3/folders/1nXrEd2XO9TgLrJy09xchVEh3A-wcvNq8



### 円錐角膜の前眼部OCT（Axial, Cornea, 両方結合したもの）から円錐角膜が進行するかしないかを識別する



### 関係者

##### メディカルスタッフ

＜Dr＞

加藤 直子 naokato@bc.iij4u.or.jp

＜ORT＞

石飛 直史 [n.ishitobi@tsukazaki-eye.net](mailto:n.ishitobi@tsukazaki-eye.net)

##### エンジニア

田邉 真生 m.tanabe@tsukazaki-eye.net 

升本 浩紀 [h.masumoto@tsukazaki-eye.net](mailto:h.masumoto@tsukazaki-eye.net)

### データセット

##### 画像データ

https://drive.google.com/open?id=1lhgEwJpMGa1n88MRbm3ifD5xeGZB9HOl

##### タグデータ

https://drive.google.com/open?id=1b2oF_ScVRtz6k_oAqFCxkvN8yq-2SELN

##### 結果

https://drive.google.com/drive/u/0/folders/11NpnH6EOlxSKuI5H-VGgHzL-1Zq8qcWy

### 解析フロー

##### データ準備および要件定義

データ準備終了

##### AI解析手法(第一回目)

[Neural Network]

眼底画像からroutine application (mode=binary classification)で解析

https://github.com/thinkout-projects/classify-image

損失関数についてはBinary-categorical CrossEntropy, OptimizerはMomentum SGD(lr= 0.0005, initial term=0.9)で解析

##### AI解析手法(第二回目)

年齢データが30%程度影響するとのこと by 加藤Dr

画像を畳み込み、Flattenしたものと、年齢をconcatenateしてから分類する。

[Outcome]

AUC, 感度、特異度

[Result 1回目]

1. Axial

AUC 0.586 (0.512-0.656)、感度 55.6 (44.7-66.0)、特異度 58.6 (51.2-65.8)

2. Cornea

AUC 0.553 (0.480-0.626)、感度 20.0 (12.3-29.8)、特異度 80.1 (73.6-85.6)

3. Both

AUC 0.534 (0.461-0.607)、感度 42.2 (31.9-53.1)、特異度 59.1 (51.7-66.3)

[Result 2回目]



