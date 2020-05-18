import tensorflow as tf
import random


@tf.function
def load_and_data_augment(path, size, ch, basic_process, extra_process):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=ch)
  # 先に追加処理
  i = random.randint(0, len(extra_process.processes) - 1)
  image = extra_process.processes[i](image)
  # その後、基本処理を全種類行う
  for i in range(len(basic_process.processes)):
    image = basic_process.processes[i](image)
  # 最後にresizeし、255で正規化して終了
  image = tf.image.resize(image, size)
  image /= 255.0  # normalize to [0,1] range
  return image


@tf.function(experimental_relax_shapes=True)
def standardized_params(par, mean, std):
  return (tf.cast(par, dtype=tf.float32) -mean)/std


@tf.function
def load_and_convert_onehot(label, classes):
  return tf.one_hot(label, classes, on_value=1.0, off_value=0.0, axis=-1)


@tf.function
def load_img(path, size, ch):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=ch)
  image = tf.image.resize(image, size)
  image /= 255.0  # normalize to [0,1] range
  return image