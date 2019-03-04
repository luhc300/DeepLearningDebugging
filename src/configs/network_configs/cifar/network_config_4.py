from src.cnn_builder import Layer

DROPOUT_PROB = 0.6
NETWORK_STRUCTURE = [Layer("conv", [3, 3, 3, 64]),
                     Layer("conv", [3, 3, 64, 64]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 64, 128]),
                     Layer("conv", [3, 3, 128,128]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 128, 256]),
                     Layer("conv", [3, 3, 256, 256]),
                     Layer("conv", [3, 3, 256, 256]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 256, 512]),
                     Layer("conv", [3, 3, 512, 512]),
                     Layer("conv", [3, 3, 512, 512]),
                     Layer("pool", []),
                     Layer("conv", [3, 3, 512, 512]),
                     Layer("conv", [3, 3, 512, 512]),
                     Layer("conv", [3, 3, 512, 512]),
                     Layer("dense", [2*2*512, 4096]),
                     Layer("dense", [4096, 4096]),
                     Layer("dense", [4096, 10])]
NETWORK_ANCHOR = -4
NETWORK_PATH = "model/cifar/model_4.ckpt"
INIT = 0.0
LEARNING_RATE = 1e-4
DISTRIBUTION_PATH = "profile/cifar/model4VGG_mdist_distribution.pkl"