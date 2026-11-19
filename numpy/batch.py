import numpy as np



def create_batches(images,labels,batch_size):


    indices=np.arange(len(images))


    np.random.shuffle(indices)


    batches=[]


    for i in range(
        0,
        len(images),
        batch_size
    ):

        batch_indices=indices[
            i:i+batch_size
        ]


        batches.append(
            (
            images[batch_indices],
            labels[batch_indices]
            )
        )


    return batches
