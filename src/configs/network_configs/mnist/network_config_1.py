from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [5, 5, 1, 32]),
                     Layer("pool", []),
                     Layer("conv", [5, 5, 32, 64]),
                     Layer("pool", []),
                     Layer("dense", [7*7*64, 1024]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [1024, 10])]
NETWORK_ANCHOR = -7
NETWORK_PATH = "model/mnist/model_1.ckpt"