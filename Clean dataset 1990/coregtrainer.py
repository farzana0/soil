import coreg
import os
import sys
import shutil
import numpy as np

class CoregTrainer:
    """
    Coreg trainer class.
    """
    def __init__(self, data_dir, results_dir, num_train, num_trials, k1=3,
                 k2=3, p1=2, p2=5, max_iters=100, pool_size=100,
                 batch_unlabeled=0, verbose=False):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.num_train = num_train
        self.num_trials = num_trials
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.batch_unlabeled = batch_unlabeled
        self.verbose = verbose
        self._remake_dir(os.path.join(self.results_dir, 'results'))
        self._setup_coreg()

    def run_trials(self):
        """
        Run multiple trials of training.
        """
        self.coreg_model.run_trials(
            self.num_train, self.num_trials, self.verbose)
        self._write_results()

    def _remake_dir(self, directory):
        """
        If directory exists, delete it and remake it, if it doesn't exist, make
        it.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    def _setup_coreg(self):
        """
        Sets up coreg regressors and load data.
        """
        self.coreg_model = coreg.Coreg(
            self.k1, self.k2, self.p1, self.p2, self.max_iters, self.pool_size)
        self.coreg_model.add_data(self.data_dir)

    def _write_results(self):
        """
        Writes results produced by experiment.
        """
        np.save(os.path.join(self.results_dir, 'results/mses1_train'),
                self.coreg_model.mses1_train)
        np.save(os.path.join(self.results_dir, 'results/mses1_test'),
                self.coreg_model.mses1_test)
        np.save(os.path.join(self.results_dir, 'results/mses2_train'),
                self.coreg_model.mses2_train)
        np.save(os.path.join(self.results_dir, 'results/mses2_test'),
                self.coreg_model.mses2_test)
        np.save(os.path.join(self.results_dir, 'results/mses_train'),
                self.coreg_model.mses_train)
        np.save(os.path.join(self.results_dir, 'results/mses_test'),
                self.coreg_model.mses_test)