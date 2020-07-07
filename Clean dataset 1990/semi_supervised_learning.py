#semi_supervised_learning.py
import os
import coregtrainer
import shutil
'''
# Coreg hyperparameters
k1: 3
k2: 3
p1: 2
p2: 5
pool_size: 100
'''

#num_train will be ignored
# small ks it was overfitting
ct = coregtrainer.CoregTrainer(data_dir=os.getcwd(), results_dir=os.getcwd(), num_train=1, num_trials=4, max_iters=9000 ,verbose=True,  pool_size=50,
                k1=12,
                 k2=15, p1=4, p2=10)
ct.run_trials()

