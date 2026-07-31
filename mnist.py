

# Download raw MNIST and convert to f32

import gzip
import os
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download(filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, filename)
    else:
        print(f"{filename} already downloaded, skipping.")


def load_images(filename):
    with gzip.open(filename, "rb") as f:
        data = f.read()
        num_images = int.from_bytes(data[4:8], "big")
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(num_images, rows * cols)
        return images


def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels


for filename in FILES.values():
    download(filename)

train_images = load_images(FILES["train_images"])
train_labels = load_labels(FILES["train_labels"])
test_images = load_images(FILES["test_images"])
test_labels = load_labels(FILES["test_labels"])

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.tofile("train_images.mat")
train_labels.tofile("train_labels.mat")
test_images.tofile("test_images.mat")
test_labels.tofile("test_labels.mat")

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)