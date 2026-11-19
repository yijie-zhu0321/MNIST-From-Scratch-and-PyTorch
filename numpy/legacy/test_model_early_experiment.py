from load_data import load_dataset
from model import LogisticRegression


train_images, train_labels = load_dataset(
    "data/MNIST/raw/train-images-idx3-ubyte.gz",
    "data/MNIST/raw/train-labels-idx1-ubyte.gz"
)

model = LogisticRegression()

image = train_images[0]
label = train_labels[0]

scores = model.forward(image)

print("Correct label:", label)
print("Scores:", scores)
print("Score shape:", scores.shape)
