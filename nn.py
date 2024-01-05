import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


activation = ["linear" #raw input
              "relu", #max(x, 0)
              "sigmoid", # sigmoid(x) = 1 / (1 + exp(-x))
              "softmax", #exp(x) / sum(exp(x))
              "softplus", #log(exp(x) + 1)
              "softsign", #x / (abs(x) + 1)
              "tanh", #sinh(x) / cosh(x)
              "hard_sigmoid", #0 if x < -2.5  || 1 if x > 2.5 || 0.2 * x + 0.5 if -2.5 <= x <= 2.5
              ]


maxEpoch = 10

inputs = Input(shape=(2,))
x = Dense(5, activation='relu')(inputs)