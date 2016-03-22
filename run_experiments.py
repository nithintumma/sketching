"""
drivers to run experiments
"""
import os 
import numpy as np

from helpers import load_matrix, write_matrix
from experiments import AlphaSketchExperiment, BatchSketchExperiment, DynamicSketchExperiment, SketchExperiment


EXPERIMENT_MATRIX_DIR = 'experiment_matrices/'

"""
TODOS:
1) DONE: Alpha experiment 
2) proof of concept for sketching (figure out hashing, write Compensative, Lossy)
3) Dynamic Sketch sizes 
"""

# random matrices 
rand_mat_1_fname = 'rand_10000_2000.txt'
# larger
rand_mat_2_fname = "NOT CREATED"
# sparse
rand_mat_3_fname = "NOT CREATED"

cifar_mat_fname = 'data_batch_1'

MATRIX = cifar_mat_fname


#TODO: get real matrices 

def alpha_experiment(mat_fname=MATRIX, l=800, alphas=None, plot=False):
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

# next experiment should be sanity test, how do we do this? 
# will need to compute sketches at different sizes using a variety of sketches 
# then plot them all on the same graph 
# theoretically works, we just need to write driver 
def compare_sketches_experiment(mat_fname=MATRIX, l=800, alphas=None, plot=False):
	# compare all sketches 
	sketch_types = {'jlt': None, 'cw':None, 'fd': None, 'pfd': {'alpha': 0.2}, 'batch-pfd': {'batch_size': 400, 'alpha': 0.2}}
	ls = np.arange(100, 1100, 100)
	exp_name = "sketch_exp_" + os.path.splitext(mat_fname)[0]
	sketch_exp = SketchExperiment(exp_name, mat_fname, ls, sketch_types=sketch_types)
	sketch_exp.run_experiment()
	sketch_exp.write_results()
	if plot:
		sketch_exp.plot_results(err=True, proj_err=True, time=True, save=True)

# dynamic experiment 
def dynamic_experiment(mat_fname=MATRIX, l1=200, l2=800, batch_size=800):
    print "Dynamic Experiment"
    mat = load_matrix(mat_fname)
    ts = np.arange(1, mat.shape[0], max(mat.shape[0]/10, 1))
    exp_name = "dynamic_exp_" + os.path.splitext(mat_fname)[0]
    dyn_exp = DynamicSketchExperiment(exp_name, mat_fname, l1, l2, batch_size, ts, randomized=False)
    dyn_exp.run_experiment()
    dyn_exp.write_results()
    if plot:
    	dyn_exp.plot_results()


# batched vs batched random 
# TODO: figure out what random algorithm fb is using, compare to what scipy has, also think about implementing one 
if __name__ == "__main__":
	#compare_sketches_experiment()
    #alpha_experiment()
    dynamic_experiment() 
