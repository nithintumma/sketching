"""
drivers to run experiments
"""
import os
import numpy as np
import cPickle as pickle

from helpers import load_matrix, write_matrix
from experiments import (AlphaSketchExperiment, BatchSketchExperiment, 
                            DynamicSketchExperiment, SketchExperiment,
                            TweakVsBatchPFDSketchExperiment)


EXPERIMENT_MATRIX_DIR = 'experiment_matrices/'

# random matrices
rand_mat_1_fname = 'rand_10000_2000.txt'
# larger
rand_mat_2_fname = "NOT CREATED"
# sparse
rand_mat_3_fname = "NOT CREATED"

cifar_mat_fname = 'data_batch_1'
small_cifar_mat_fname = 'small_data_batch_1'
large_cifar_mat_fname = 'cifar_data'
MATRIX = large_cifar_mat_fname

def alpha_experiment(mat_fname=MATRIX, l=200, alphas=None, plot=True):
    """
    changing alpha (FD param), fixed sketch size
    """
    print "Alpha Experiment"
    if alphas is None:
        alphas = np.arange(0.1, 1.1, 0.1)
    print "Testing alphas: ", alphas
    exp_name = "pfd_alpha_exp_" + os.path.splitext(mat_fname)[0]
    pfd_exp = AlphaSketchExperiment(exp_name, mat_fname, l, alphas)
    pfd_exp.run_experiment()
    print "Ran experiments"
    pfd_exp.write_results()
    if plot:
        pfd_exp.plot_results(err=True, proj_err=True, time=True, save=True)

# we are initing an empty class, seems wasteful 
def plot_sketches_results(exp_name, mat_fname, results_fname, sketch_types={}):
    mat_name = os.path.splitext(mat_fname)[0]
    results_path = os.path.join("experiments", exp_name, mat_fname, results_fname)
    with open(results_path, "rb") as pfile: 
        results = pickle.load(pfile)
    ls = np.sort(results['pfd'].keys())
    print ls
    sk_obj = SketchExperiment(exp_name, mat_fname, ls, sketch_types)
    sk_obj.results = results
    sk_obj.computed_results=True
    sk_obj.plot_results()

def compare_sketches_experiment(mat_fname=MATRIX, l_low=100, l_high=400, plot=True):
    print "Sketching Experiment"
    # compare all sketches 
    sketch_types = {'jlt': None, 'cw':None, 'fd': None, 'pfd': {'alpha': 0.2}, 'batch-pfd': {'batch_size': 400, 'alpha': 0.2}}
    ls = np.arange(l_low, l_high+100, (l_high - l_low)/15)
    exp_name = "sketch_exp_" + os.path.splitext(mat_fname)[0]
    sketch_exp = SketchExperiment(exp_name, mat_fname, ls, sketch_types=sketch_types)
    sketch_exp.run_experiment()
    sketch_exp.write_results()
    if plot:
        sketch_exp.plot_results(err=True, proj_err=True, time=True, save=True)

# dynamic experiment 
def dynamic_experiment(mat_fname=MATRIX, l1=320, l2=350, batch_size=400, plot=True):
    print "Dynamic Experiment"
    mat = load_matrix(mat_fname)
    # changepoints 
    ts = np.arange(1, mat.shape[0], max(mat.shape[0]/10, 1))
    exp_name = "dynamic_exp_" + os.path.splitext(mat_fname)[0]
    dyn_exp = DynamicSketchExperiment(exp_name, mat_fname, l1, l2, batch_size, ts, randomized=False)
    dyn_exp.run_experiment()
    dyn_exp.write_results()
    if plot:
        dyn_exp.plot_results()

# batched vs batched random 
def tweak_vs_batched_experiment(mat_fname=MATRIX, l=200, alphas=np.arange(0.1, 1.1, 0.1), fast=False):
    # what is init signature? 
    print "Tweak vs Batched Experiment"
    #mat = load_matrix(mat_fname)
    #def __init__(self, exp_name, mat_fname, l, alphas, runs=3, randomized=False):
    exp_name = "tweak_batch_exp_" + os.path.splitext(mat_fname)[0]
    exp = TweakVsBatchPFDSketchExperiment(exp_name, mat_fname, l, alphas, runs=3, fast=fast)
    exp.run_experiment()
    exp.write_results()

def completed_experiments():
    dynamic_experiment(mat_fname=large_cifar_mat_fname,
                    l1=200,
                    l2=300,
                    batch_size=300,
                    plot=False) 



# TODO: figure out what random algorithm fb is using, compare to what scipy has, also think about implementing one 
if __name__ == "__main__":
    tweak_vs_batched_experiment(mat_fname=small_cifar_mat_fname, l=200, alphas=np.arange(0.1, 1.1, 0.1), fast=False)
