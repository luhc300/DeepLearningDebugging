from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [3, 3, 1, 32]),
                     Layer("conv", [3, 3, 32, 32]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 32, 64]),
                     Layer("conv", [3, 3, 64, 64]),
                     Layer("pool", []),
                     Layer("dense", [7*7*64, 200]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [200, 200]),
                     Layer("dense", [200, 10])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/mnist/model_4.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-3