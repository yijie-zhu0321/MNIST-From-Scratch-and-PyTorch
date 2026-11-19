"""Fast checks using synthetic inputs; does not train on or download MNIST."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numpy"))
from model import NeuralNetwork
from backward import backward
from loss import cross_entropy_loss
from optimizer import SGD

class TestNumPyModel(unittest.TestCase):
    def test_forward_and_gradients(self):
        np.random.seed(11)
        model = NeuralNetwork()
        image = np.random.rand(784)
        label = 4
        scores = model.forward(image)
        self.assertEqual(scores.shape, (10,))
        grads = backward(model, image, label)
        for name in ("W1", "b1", "W2", "b2"):
            self.assertEqual(grads[name].shape, getattr(model, name).shape)
        eps = 1e-5
        for name, index in (("W1", (2, 3)), ("W2", (3, 4)), ("b1", (3,)), ("b2", (4,))):
            param = getattr(model, name)
            original = param[index]
            param[index] = original + eps
            loss_plus = cross_entropy_loss(model.forward(image), label)
            param[index] = original - eps
            loss_minus = cross_entropy_loss(model.forward(image), label)
            param[index] = original
            numeric = (loss_plus - loss_minus) / (2 * eps)
            self.assertAlmostEqual(numeric, grads[name][index], delta=1e-5)
        initial = cross_entropy_loss(model.forward(image), label)
        SGD(lr=0.01).step(model, grads)
        after = cross_entropy_loss(model.forward(image), label)
        self.assertLess(after, initial)

class TestPyTorchModel(unittest.TestCase):
    def test_forward_shape(self):
        sys.path.insert(0, str(ROOT / "pytorch"))
        from mnist import MNISTMLP
        model = MNISTMLP()
        self.assertEqual(tuple(model(torch.rand(2, 1, 28, 28)).shape), (2, 10))

if __name__ == "__main__":
    unittest.main()
