from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [3, 3, 3, 32]),
                     Layer("conv", [3, 3, 32, 32]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 32, 64]),
                     Layer("conv", [3, 3, 64, 64]),
                     Layer("pool", []),
                     Layer("dense", [8*8*64, 1024]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [1024, 1024]),
                     Layer("dense", [1024, 10])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/cifar/model_3.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-3
DISTRIBUTION_PATH = "profile/cifar/model3CNN_mdist_distribution.pkl"