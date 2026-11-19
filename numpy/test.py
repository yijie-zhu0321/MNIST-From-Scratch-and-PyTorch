import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from load_data import load_images, load_labels
from model import NeuralNetwork



# load test data

images = load_images(
    BASE_DIR / "data/MNIST/raw/t10k-images-idx3-ubyte.gz"
)

labels = load_labels(
    BASE_DIR / "data/MNIST/raw/t10k-labels-idx1-ubyte.gz"
)


images = images.reshape(-1,784)

images = images.astype(float)/255



# create model

model = NeuralNetwork()



# load weights

data = np.load(
    BASE_DIR / "mnist_model.npz"
)


model.W1 = data["W1"]
model.b1 = data["b1"]

model.W2 = data["W2"]
model.b2 = data["b2"]


print("Model loaded!")



correct = 0


for image,label in zip(images,labels):


    scores = model.forward(image)


    prediction = np.argmax(scores)


    if prediction == label:

        correct += 1



print("----------------------")
print("Correct:",correct)
print("Total:",len(labels))
print(
    "Accuracy:",
    correct/len(labels)
)
print("----------------------")
