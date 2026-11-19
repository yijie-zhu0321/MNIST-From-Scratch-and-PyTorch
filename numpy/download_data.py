"""Download the four public MNIST gzip archives used by the NumPy exercises."""
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = DATA_DIR / name
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(BASE_URL + name, destination)
    print(f"MNIST archives ready in {DATA_DIR}")

if __name__ == "__main__":
    main()
