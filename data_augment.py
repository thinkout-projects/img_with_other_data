import tensorflow as tf
import cv2
import numpy as np


class BasicProcess:
    processes = []
    rotate = True
    rotate_range = 15
    shift = True
    width_shift_range = 0.05
    height_shift_range = 0.05
    shear = True
    shear_range = 0.1
    zoom = True
    zoom_range = 0.1

    def __init__(self, rotate=True, shift=True, shear=True, zoom=True, h_flip=True, v_flip=True):
        self.processes = []
        self.processes.append(self.processes_in_tf)
        self.rotate = rotate
        self.shift = shift
        self.shear = shear
        self.zoom = zoom
        if h_flip:
            self.processes.append(self.flip_h)
        if v_flip:
            self.processes.append(self.flip_v)

    def processes_in_numpy(self, img_tf):
        img = img_tf.numpy()
        if self.rotate:
            img = tf.keras.preprocessing.image.random_rotation(img, self.rotate_range, 0, 1, 2)
        if self.shift:
            img = tf.keras.preprocessing.image.random_shift(img, self.width_shift_range, self.height_shift_range, 0, 1,
                                                            2)
        if self.shear:
            img = tf.keras.preprocessing.image.random_shear(img, self.shear_range, 0, 1, 2)
        if self.zoom:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range, 2)
            img = tf.keras.preprocessing.image.random_zoom(img, zoom_factor, 0, 1, 2)
        return img

    @tf.function
    def processes_in_tf(self, image):
        x = tf.py_function(self.processes_in_numpy, [image], [tf.uint8])
        X = x[0]
        X.set_shape(image.shape)
        return X

    @tf.function
    def flip_h(self, image):
        return tf.image.random_flip_left_right(image)

    @tf.function
    def flip_v(self, image):
        return tf.image.random_flip_up_down(image)


class ExtraProcess:
    processes = []
    contrast_factor_lower = 0.2
    contrast_factor_upper = 1.8
    gamma_lower = 0.75
    gamma_upper = 1.5
    filter_kernel_size = 3
    gaussian_noise_std = 30
    s_vs_p = 0.5
    amount = 0.004

    def __init__(self, contrast=True, gamma=True, gaussian_blur=True,
                 histogram_equalization=False, gaussian_noise=True, saltpepper_noise=True):
        self.processes = []
        self.processes.append(self.nothing)
        if contrast:
            self.processes.append(self.adjust_contrast)
        if gamma:
            self.processes.append(self.adjust_gamma)
        if gaussian_blur:
            self.processes.append(self.blur)
        if histogram_equalization:
            self.processes.append(self.hist)
        if gaussian_noise:
            self.processes.append(self.addGaussianNoise)
        if saltpepper_noise:
            self.processes.append(self.addSaltPepperNoise)

    @tf.function
    def nothing(self, image):
        return image

    @tf.function
    def adjust_contrast(self, image):
        return tf.image.random_contrast(image, self.contrast_factor_lower, self.contrast_factor_upper, seed=1)

    @tf.function
    def adjust_gamma(self, image):
        gamma = \
            tf.random.uniform([1], minval=self.gamma_lower, maxval=self.gamma_upper, dtype=tf.dtypes.float32, seed=1)[0]
        return tf.image.adjust_gamma(image, gamma=gamma)

    def blur_in_numpy(self, img_tf):
        img = img_tf.numpy()
        img = cv2.blur(img, (self.filter_kernel_size, self.filter_kernel_size))
        return img

    @tf.function
    def blur(self, image):
        x = tf.py_function(self.blur_in_numpy, [image], [tf.uint8])
        X = x[0]
        X.set_shape(image.shape)
        return X

    def hist_in_numpy(self, img_tf):
        img = img_tf.numpy()
        RGB = cv2.split(img)
        for i in range(3):
            cv2.equalizeHist(RGB[i])
        img_hist = cv2.merge([RGB[0], RGB[1], RGB[2]])
        return img_hist

    @tf.function
    def hist(self, image):
        x = tf.py_function(self.hist_in_numpy, [image], [tf.uint8])
        X = x[0]
        X.set_shape(image.shape)
        return X

    def addGaussianNoise_in_numpy(self, img_tf):
        img = img_tf.numpy()
        gauss = np.random.normal(0, self.gaussian_noise_std, img.shape)
        noisy = img + gauss
        return noisy

    @tf.function
    def addGaussianNoise(self, image):
        x = tf.py_function(self.addGaussianNoise_in_numpy, [image], [tf.uint8])
        X = x[0]
        X.set_shape(image.shape)
        return X

    def addSaltPepperNoise_in_numpy(self, img_tf):
        img = img_tf.numpy()
        out = img.copy()
        num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[tuple(coords[:-1])] = (255, 255, 255)
        # Pepper mode
        num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[tuple(coords[:-1])] = (0, 0, 0)
        return out

    @tf.function
    def addSaltPepperNoise(self, image):
        x = tf.py_function(self.addSaltPepperNoise_in_numpy, [image], [tf.uint8])
        X = x[0]
        X.set_shape(image.shape)
        return X

