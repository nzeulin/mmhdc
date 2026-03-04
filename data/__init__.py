import os
import numpy as np
from struct import unpack

def load_mnist(dataset: str = "mnist"):
    def load_mnist_images(filename: str) -> np.ndarray:
        with open(filename, 'rb') as f:
            f.read(4)                       # magic number, not needed
            N = unpack('>i', f.read(4))[0]  # number of items
            image_size = 28
            f.read(8)                       # image size (repeated 2 times), 28 is default
            return np.frombuffer(f.read(image_size * image_size * N), dtype='B').reshape(N, image_size, image_size).copy()

    def load_mnist_labels(filename: str) -> np.ndarray:
        with open(filename, 'rb') as f:
            f.read(4)                       # magic number
            N = unpack('>i', f.read(4))[0]  # number of items
            return np.frombuffer(f.read(N), dtype='B').reshape(N).copy()

    cwd = os.getcwd()
    X_train = load_mnist_images(cwd + '/data/{}/train-images-idx3-ubyte'.format(dataset)).astype(np.float32)
    y_train = load_mnist_labels(cwd + '/data/{}/train-labels-idx1-ubyte'.format(dataset)).astype(np.int64)
    X_test = load_mnist_images(cwd + '/data/{}/t10k-images-idx3-ubyte'.format(dataset)).astype(np.float32)
    y_test = load_mnist_labels(cwd + '/data/{}/t10k-labels-idx1-ubyte'.format(dataset)).astype(np.int64)

    return X_train, y_train, X_test, y_test
