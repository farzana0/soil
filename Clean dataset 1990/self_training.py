# self_training.py


import numpy as np
data = np.load('results/mses_test.npy')
y = np.load('results/mses_train.npy')
x = np.load('results/mses1_train.npy')
y = np.load('results/mses1_test.npy')
x = np.load('results/mses2_train.npy')
y = np.load('results/mses2_test.npy')

print(data)
print(y)
print(data)
print(y)