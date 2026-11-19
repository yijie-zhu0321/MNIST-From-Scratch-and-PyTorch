import numpy as np
import gzip
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_images(filename):
    """
    Load MNIST images from a gzip-compressed IDX file.
    Returns shape: (N, 28, 28)
    """
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=16
        )

    images = data.reshape(-1, 28, 28)
    return images


def load_labels(filename):
    """
    Load MNIST labels from a gzip-compressed IDX file.
    Returns shape: (N,)
    """
    with gzip.open(filename, "rb") as f:
        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=8
        )

    return labels


def normalize_images(images):
    """
    Convert pixel values from 0-255 to 0-1.
    """
    images = images.astype(np.float32)
    images = images / 255.0
    return images


def flatten_images(images):
    """
    Convert images from (N, 28, 28) to (N, 784).
    """
    images = images.reshape(images.shape[0], -1)
    return images


def load_dataset(image_file, label_file):
    """
    Load and preprocess the dataset.
    """
    images = load_images(image_file)
    labels = load_labels(label_file)

    images = normalize_images(images)
    images = flatten_images(images)

    return images, labels


if __name__ == "__main__":

    train_images, train_labels = load_dataset(
        BASE_DIR / "data/MNIST/raw/train-images-idx3-ubyte.gz",
        BASE_DIR / "data/MNIST/raw/train-labels-idx1-ubyte.gz"
    )

    print("Images shape:", train_images.shape)
    print("Labels shape:", train_labels.shape)

    print("First label:", train_labels[0])
    print("First image first 10 pixels:")
    print(train_images[0][:10])
