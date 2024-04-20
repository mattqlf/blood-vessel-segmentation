import numpy as np
class Config:
    image_size = 1024
    lr = 8e-5
    epochs = 40
    folds = 5
    model = "Unet"
    encoder = "resnet50"
    threshold = 0.5
    clip = (3, -3)
    loss = "focal"
    show_fig = False
    debug = False
    ref = np.load("./reference.npy")