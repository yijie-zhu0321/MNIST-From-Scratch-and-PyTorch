import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
np.random.seed(42)

from load_data import load_images, load_labels
from model import NeuralNetwork
from loss import cross_entropy_loss
from backward import backward
from optimizer import SGD



# load data

train_images = load_images(
    BASE_DIR / "data/MNIST/raw/train-images-idx3-ubyte.gz"
)

train_labels = load_labels(
    BASE_DIR / "data/MNIST/raw/train-labels-idx1-ubyte.gz"
)


# preprocess

train_images = train_images.reshape(-1,784)

train_images = train_images.astype(np.float32) / 255.0



# create model

model = NeuralNetwork()


optimizer = SGD(
    lr=0.01
)



epochs = 5



for epoch in range(epochs):


    total_loss = 0


    for i in range(len(train_images)):


        image = train_images[i]

        label = train_labels[i]


        # forward

        scores = model.forward(image)


        # loss

        loss = cross_entropy_loss(
            scores,
            label
        )


        total_loss += loss



        # backward

        gradients = backward(
            model,
            image,
            label
        )


        # update

        optimizer.step(
            model,
            gradients
        )



    print(
        "Epoch:",
        epoch+1,
        "Loss:",
        total_loss / len(train_images)
    )



# save new model

np.savez(
    BASE_DIR / "mnist_model.npz",

    W1=model.W1,
    b1=model.b1,

    W2=model.W2,
    b2=model.b2
)


print("Model saved!")
