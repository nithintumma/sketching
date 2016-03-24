import numpy as np 
import matplotlib.pyplot as plt 
import cPickle as pickle 
import os
from experiments import DynamicSketchExperiment 

# get this from the experiments fun
def plot_dynamic_sketch_experiment(results_fname, save=True):
	"""
	independent variable: Changepoint
	"""
	# plot the three graphs stacked, sharing the x axis (Changepoint)
	with open(results_fname, "rb") as f:
		results = pickle.load(f)

	# sort the integer keys of results 
	changepoints = np.sort(filter(lambda x: isinstance(x, (int, long)), results.keys()))
	xlims = [0, changepoints[-1]]
	# how do we stack the plots on top of each other in python?
	# the three plots here are err, proj_err, 
	fig, ax_arr = plt.subplots(3, sharex=True, figsize=(10, 30))
	plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))

	# covariance errors
	cov_errs = [results[t]['err'] for t in changepoints]
	l1_cov_err = results['l1']['err']
	l2_cov_err = results['l2']['err']
	ax_arr[0].plot(changepoints, cov_errs, '-o', label='cov err')
	ax_arr[0].hlines(l1_cov_err, xlims[0], xlims[1], "r", label='l1 cov err')
	ax_arr[0].hlines(l2_cov_err, xlims[0], xlims[1], "r", label='l2 cov err')
	ax_arr[0].set_ylabel("Covariance Reconstruction Error")

	# projection errors
	proj_errs = [results[t]['proj_err'] for t in changepoints]
	l1_proj_err = results['l1']['proj_err']
	l2_proj_err = results['l2']['proj_err']
	ax_arr[1].plot(changepoints, proj_errs, '-o', label='proj err')
	ax_arr[1].hlines(l1_proj_err, xlims[0], xlims[1], "r", label='l1 proj err')
	ax_arr[1].hlines(l2_proj_err, xlims[0], xlims[1], "r", label='l2 proj err')
	ax_arr[1].set_ylabel("Projection Error")

	# l hat bounds
	l1 = results['l1_size']
	l2 = results['l2_size']
	l_bounds = [results[t]['l_bound'] for t in changepoints]
	ax_arr[2].plot(changepoints, l_bounds, '-o', label='L Bound')
	ax_arr[2].hlines(l1, xlims[0], xlims[1], "r", label='l1')
	ax_arr[2].hlines(l2, xlims[0], xlims[1], "r", label='l2')
	ax_arr[2].set_ylabel("Bound on Realized L")

	# set up plots 
	plt.tight_layout()
	for ax in ax_arr:
		ax.grid()

	if save: 
		fig.savefig(os.path.join(results['exp_dir'], "results.png"))
	else:
		fig.show()

def plot_batched_sketch_experiment(results_fname, results_rand_fname, save=True):
	pass

if __name__ == "__main__":
	#fname = "experiments/dynamic_exp_data_batch_1/data_batch_1/results.p"
	fname = "temp.p"
	plot_dynamic_sketch_experiment(fname, save=True)
