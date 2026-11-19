import numpy as np


class NeuralNetwork:

    def __init__(self):

        # Layer 1
        self.W1 = np.random.randn(784,128) * 0.01
        self.b1 = np.zeros(128)


        # Layer 2
        self.W2 = np.random.randn(128,10) * 0.01
        self.b2 = np.zeros(10)



    def relu(self,x):

        return np.maximum(0,x)



    def forward(self,image):

        image = image.flatten()


        # First layer
        self.z1 = image @ self.W1 + self.b1


        # ReLU
        self.a1 = self.relu(self.z1)


        # Second layer
        scores = self.a1 @ self.W2 + self.b2


        return scores
