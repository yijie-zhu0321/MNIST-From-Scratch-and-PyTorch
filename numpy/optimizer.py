class SGD:


    def __init__(self,lr=0.01):

        self.lr=lr



    def step(self,model,gradients):


        model.W1 -= self.lr * gradients["W1"]
        model.b1 -= self.lr * gradients["b1"]


        model.W2 -= self.lr * gradients["W2"]
        model.b2 -= self.lr * gradients["b2"]
