"""
drivers to run experiments
"""
import os
import numpy as np
import cPickle as pickle

from helpers import load_matrix, write_matrix
from experiments import (AlphaSketchExperiment, BatchSketchExperiment, 
                            DynamicSketchExperiment, SketchExperiment,
                            TweakVsBatchPFDSketchExperiment, 
                            BatchRandomPFDSketchExperiment, 
                            ParallelPFDSketchExperiment)


EXPERIMENT_MATRIX_DIR = 'experiment_matrices/'

# random matrices
rand_mat_1_fname = 'rand_10000_2000.txt'
# CIFAR matrices
cifar_mat_fname = 'data_batch_1'
small_cifar_mat_fname = 'small_data_batch_1'
med_cifar_mat_fname = 'cifar_data'
large_cifar_mat_fname = 'large_cifar_data'
# sparse matrices
sparse_mat_fname = 'ESOC.mtx'
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


# tweak(Fast) vs batched PFD 
def tweak_vs_batched_experiment(mat_fname=MATRIX, l=200, alphas=np.arange(0.1, 1.1, 0.1), fast=False):
    # what is init signature? 
    print "Tweak vs Batched Experiment"
    #mat = load_matrix(mat_fname)
    #def __init__(self, exp_name, mat_fname, l, alphas, runs=3, randomized=False):
    exp_name = "tweak_batch_exp_" + os.path.splitext(mat_fname)[0]
    exp = TweakVsBatchPFDSketchExperiment(exp_name, mat_fname, l, alphas, runs=3, fast=fast)
    exp.run_experiment()
    exp.write_results()

# batched vs batched random (changing batch size) 
def rand_batch_experiment(mat_fname=MATRIX, l=200, alpha=0.2, batch_sizes=None, runs=2):
    print "Randomized Batched Experiment"
    exp_name =  "rand_batch_exp_" + os.path.splitext(mat_fname)[0]
    if batch_sizes is None:
        batch_sizes = np.arange(l/5, 5*l, l/2)
    exp = BatchRandomPFDSketchExperiment(exp_name, mat_fname, l, alpha, batch_sizes, runs=runs)
    exp.run_experiment()
    exp.write_results()

def run_parallel_experiment(mat_fname, l, alpha, batch_size, processors, runs):
    print "Parallel Experiment"
    print "Testing: ", processors
    exp_name = 'parallel_exp_' + os.path.splitext(mat_fname)[0]
    if mat_fname[-3:] == 'mtx':
        sparse = True
        print "Sparse Matrix: ", mat_fname
    else:
        sparse=False
    exp = ParallelPFDSketchExperiment(exp_name, mat_fname, l, alpha, 
                                        batch_size, 
                                        processors=processors, 
                                        runs=2, sparse=sparse)
    exp.run_experiment()
    exp.write_results()
    print "FINISHED"

def completed_experiments():
    dynamic_experiment(mat_fname=large_cifar_mat_fname,
                    l1=200,
                    l2=300,
                    batch_size=300,
                    plot=False) 
    mat_fname = sparse_mat_fname
    l = 200
    alpha = 0.2
    batch_size = 2 * l
    processors = [1, 2, 4, 8, 16, 32]
    runs = 2
    run_parallel_experiment(mat_fname, l, alpha, batch_size, processors, runs)
    # just around so I don't have to rewrite every time 
    mat_fname = med_cifar_mat_fname
    l1 = 200
    l2 = 300
    alpha = 0.2
    batch_size = 400
    dynamic_experiment(mat_fname=mat_fname, l1=l1, l2=l2, 
                        batch_size=400, plot=False)

if __name__ == "__main__":
    mat_fname = sparse_mat_fname
    l = 200
    alpha = 0.2
    # should we do a larger batch size for randomization? 
    batch_size = 2 * l
    processors = [2, 4, 8, 16]
    runs = 1
    run_parallel_experiment(mat_fname, l, alpha, batch_size, processors, runs)
    # when will randomization start to help out a lot? is it only for sparse
    # matrices? and if so, why is that? can we take advantage of it using scipy
    # or not really? I dont know but I kind of want to for this project
