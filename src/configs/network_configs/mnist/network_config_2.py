from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("conv", [5, 5, 1, 16]),
                     Layer("pool", []),
                     Layer("dense", [14*14*16, 1024]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [1024, 10])]
NETWORK_ANCHOR = -4
NETWORK_PATH = "model/mnist/model_2.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-3
DISTRIBUTION_PATH = "profile/mnist/model2_mdist_distribution.pkl"