import numpy as np 
import matplotlib.pyplot as plt 
import cPickle as pickle 
import os
from experiments import DynamicSketchExperiment 

# make this just take in result fnames 
def plot_batched_experiments(svd_batch_exp, rand_batch_exp, err=True, proj_err=True, time=True, save=True):
    results = {"random": rand_batch_exp.results, "svd": svd_batch_exp.results}
    if err: 
        fig = plt.figure()
        plt.grid()
        rand_errs = [results['random'][b]['err'] for b in rand_batch_exp.batch_sizes]
        svd_errs = [results['svd'][b]['err'] for b in svd_batch_exp.batch_sizes]
        # we should make sure that we tested the same batch sizes
        plt.plot(rand_batch_exp.batch_sizes, rand_errs, '-o', label='Randomized')
        plt.plot(svd_batch_exp.batch_sizes, svd_errs, '-o', label='SVD')
        plt.xlabel("Sketch Size")
        plt.ylabel("Covariance Reconstruction Error")
        plt.legend(loc="best")
        if save:
            fig.savefig(os.path.join(svd_batch_exp.exp_dir, "err_plt.png"))
        else:
            fig.show()
    if proj_err: 
        fig = plt.figure()
        plt.grid()
        rand_errs = [results['random'][b]['proj_err'] for b in rand_batch_exp.batch_sizes]
        svd_errs = [results['svd'][b]['proj_err'] for b in svd_batch_exp.batch_sizes]
        # we should make sure that we tested the same batch sizes
        plt.plot(rand_batch_exp.batch_sizes, rand_errs, '-o', label='Randomized')
        plt.plot(svd_batch_exp.batch_sizes, svd_errs, '-o', label='SVD')
        plt.xlabel("Sketch Size")
        plt.ylabel("Projection Error")
        plt.legend(loc="best")
        if save:
            fig.savefig(os.path.join(svd_batch_exp.exp_dir, "proj_err_plt.png"))
        else:
            fig.show()
    if time: 
        fig = plt.figure()
        plt.grid()
        rand_times = [results['random'][b]['time'] for b in rand_batch_exp.batch_sizes]
        svd_times = [results['svd'][b]['time'] for b in svd_batch_exp.batch_sizes]
        # we should make sure that we tested the same batch sizes
        plt.plot(rand_batch_exp.batch_sizes, rand_times, '-o', label='Randomized')
        plt.plot(svd_batch_exp.batch_sizes, svd_times, '-o', label='SVD')
        plt.xlabel("Sketch Size")
        plt.ylabel("Runtime (s)")
        plt.legend(loc="best")
        if save:
            fig.savefig(os.path.join(svd_batch_exp.exp_dir, "time_plt.png"))
        else:
            fig.show()

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
	ax_arr[2].set_ylim(max(l1/1.1, 0), l2 * 1.1)
	# set up plots 
	plt.tight_layout()
	for ax in ax_arr:
		ax.grid()

	if save: 
		fig.savefig(os.path.join(results['exp_dir'], "results.png"))
	else:
		fig.show()

def plot_tweak_batched_experiment(results_fname, save=True):
	# plot alphas vs runtime, cov_err, and reconstruction error 
	with open(results_fname, "rb") as f:
		results = pickle.load(f)
	alphas = np.sort(filter(lambda x: isinstance(x, (float)), results['tweak'].keys()))
	print "Alphas ", alphas
	xlims = [0, alphas[-1] + 0.1]
	fig, ax_arr = plt.subplots(3, sharex=True, figsize=(10, 30))
	plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
	

	# plot runtimes
	batch_times = [results['batch'][a]['time'] for a in alphas]
	tweak_times = [results['tweak'][a]['time'] for a in alphas]
	ax_arr[0].plot(alphas, batch_times, '-o', label='BPFD Runtime')
	ax_arr[0].plot(alphas, tweak_times, '-x', label='PFD Runtime')
	ax_arr[0].set_ylabel("Runtime (s)")
	ax_arr[0].legend(loc='best')


	# plot covariance errors
	batch_cov_errs = [results['batch'][a]['err'] for a in alphas]
	tweak_cov_errs = [results['tweak'][a]['err'] for a in alphas]
	ax_arr[1].plot(alphas, batch_cov_errs, '-o', label='BPFD Cov Err')
	ax_arr[1].plot(alphas, tweak_cov_errs, '-x', label='PFD Cov Err')
	ax_arr[1].set_ylabel("Covariance Reconstruction Error")
	ax_arr[1].legend(loc='best')
	# plot projection errors
	batch_proj_errs = [results['batch'][a]['proj_err'] for a in alphas]
	tweak_proj_errs = [results['tweak'][a]['proj_err'] for a in alphas]
	ax_arr[2].plot(alphas, batch_proj_errs, '-o', label='BPFD Proj Err')
	ax_arr[2].plot(alphas, tweak_proj_errs, '-x', label='PFD Proj Err')
	ax_arr[2].set_ylabel("Projection Error")
	ax_arr[2].legend(loc='best')

	plt.tight_layout()
	for ax in ax_arr[:2]:
		ax.grid()
		ax.set_yscale('log')

	# save plot 
	if save:
		fig.savefig(os.path.join(results['exp_dir'], "results.png"))
	else:
		fig.show()

# do we also want to create a function for alpha experiment? 
# we have the data here on alpha choice tho! cause we also plot errors 

def plot_batched_sketch_experiment(results_fname, results_rand_fname, save=True):
	pass

if __name__ == "__main__":
	fname = "experiments/tweak_batch_exp_small_data_batch_1/small_data_batch_1/results.p"
	plot_tweak_batched_experiment(fname, save=True)
