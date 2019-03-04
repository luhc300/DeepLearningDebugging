from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [3, 3, 1, 32]),
                     Layer("conv", [3, 3, 32, 32]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 32, 64]),
                     Layer("conv", [3, 3, 64, 64]),
                     Layer("pool", []),
                     Layer("dense", [7*7*64, 200]),
                     Layer("dense", [200, 200]),
                     Layer("dense", [200, 2])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/mnist/model_5.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-3
DISTRIBUTION_PATH = "profile/mnist/model5_mdist_distribution.pkl"