import numpy as np



def backward(model,image,label):


    # Forward values already stored

    x = image.flatten()


    # ----------------
    # softmax gradient
    # ----------------

    scores = model.a1 @ model.W2 + model.b2


    exp_scores = np.exp(
        scores - np.max(scores)
    )

    probs = exp_scores / np.sum(exp_scores)



    dz2 = probs.copy()

    dz2[label] -= 1



    # ----------------
    # Layer 2 gradient
    # ----------------


    dW2 = np.outer(
        model.a1,
        dz2
    )

    db2 = dz2



    # ----------------
    # Back through layer 1
    # ----------------


    da1 = model.W2 @ dz2



    # ReLU backward

    dz1 = da1.copy()

    dz1[model.z1 <= 0] = 0



    # Layer 1 gradient


    dW1 = np.outer(
        x,
        dz1
    )

    db1 = dz1



    gradients = {

        "W1":dW1,
        "b1":db1,

        "W2":dW2,
        "b2":db2

    }


    return gradients
