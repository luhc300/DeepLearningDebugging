from src.cnn_builder import Layer

DROPOUT_PROB = 1
NETWORK_STRUCTURE = [Layer("dense", [28*28*1, 1024]),
                     Layer("dropout", [DROPOUT_PROB]),
                     Layer("dense", [1024, 10])]
NETWORK_ANCHOR = -2
NETWORK_PATH = "model/mnist/model_3.ckpt"
INIT = 1e-1
LEARNING_RATE = 1e-2