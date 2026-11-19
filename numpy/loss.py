import numpy as np


def softmax(scores):

    exp_scores = np.exp(scores - np.max(scores))

    probabilities = exp_scores / np.sum(exp_scores)

    return probabilities



def cross_entropy_loss(scores, label):

    probabilities = softmax(scores)

    loss = -np.log(probabilities[label] + 1e-12)
    return loss


if __name__ == "__main__":

    scores = np.array([2.0, 1.0, 0.1])

    label = 0

    loss = cross_entropy_loss(scores, label)

    print("Loss:", loss)
