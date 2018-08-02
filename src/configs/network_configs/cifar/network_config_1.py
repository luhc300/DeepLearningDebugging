from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [5, 5, 3, 32]),
                     Layer("pool", []),
                     Layer("conv", [5, 5, 32, 64]),
                     Layer("pool", []),
                     Layer("dense", [8*8*64, 1024]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [1024, 10])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/cifar/model_1.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-3