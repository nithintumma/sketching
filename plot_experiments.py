import numpy as np 
import matplotlib.pyplot as plt 
import cPickle as pickle 
import os
from experiments import DynamicSketchExperiment 

FIG_SIZE = (7, 15)

def plot_lbounds_err(results_fname):
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    fig, ax_arr = plt.subplots(2, figsize=(8, 10))
    # l hat bounds
    changepoints = np.sort(filter(lambda x: isinstance(x, (int, long)), results.keys()))
    xlims = [0, changepoints[-1]]
    l1 = results['l1_size']
    l2 = results['l2_size']
    l_bounds = [results[t]['l_bound'] for t in changepoints]
    errs = [0.000014, 0.000021, 0.000029, 0.000038, 0.000046, 0.000054, 0.000063, 0.000071, 0.00008, 0.000088]
    ax_arr[0].plot(changepoints, errs, '-o', label='cov err')
    ax_arr[0].hlines(0.00009, xlims[0], xlims[1], linestyle='--', label='l1 bound')
    ax_arr[0].hlines(0.000014, xlims[0], xlims[1], linestyle='-', label='l2 bound')
    ax_arr[0].set_ylabel("Covariance Error")
    #ax_arr[0].set_xlabel("Changepoint")
    ax_arr[0].set_xlim(xlims)

    ax_arr[1].plot(changepoints, l_bounds, '-o', label='l bound')
    ax_arr[1].hlines(l1, xlims[0], xlims[1], linestyle='--', label='l1')
    ax_arr[1].hlines(l2, xlims[0], xlims[1], linestyle='-', label='l2')
    ax_arr[1].set_ylabel("Bound on Realized L")
    ax_arr[1].set_xlabel("Changepoint")
    ax_arr[1].set_xlim(xlims)
    ax_arr[1].set_ylim((100, 700))

    for ax in ax_arr:
        ax.grid()
        ax.legend(loc='best')
    plt.xlim(xlims)
    
    fig.savefig(os.path.join(results['exp_dir'], "new_results.png"))


def plot_lbounds(results_fname):
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    fig = plt.figure()
    # l hat bounds
    changepoints = np.sort(filter(lambda x: isinstance(x, (int, long)), results.keys()))
    xlims = [0, changepoints[-1]]
    l1 = results['l1_size']
    l2 = results['l2_size']
    l_bounds = [results[t]['l_bound'] for t in changepoints]
    plt.plot(changepoints, l_bounds, '-o', label='l Bound')
    plt.hlines(l1, xlims[0], xlims[1], linestyle='--', label='l1')
    plt.hlines(l2, xlims[0], xlims[1], linestyle='-', label='l2')
    plt.ylabel("Bound on Realized L")
    plt.xlabel("Changepoint")
    plt.xlim(xlims)
    plt.ylim(max(l1/1.1, 0), l2 * 1.1)
    plt.legend(loc='best')
    plt.grid()
    fig.savefig(os.path.join(results['exp_dir'], "l_results.png"))

# changepoint vs 
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
    fig, ax_arr = plt.subplots(3, sharex=True, figsize=FIG_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))

    # covariance errors
    cov_errs = [results[t]['err'] for t in changepoints]
    l1_cov_err = results['l1']['err']
    l2_cov_err = results['l2']['err']
    ax_arr[0].plot(changepoints, cov_errs, '-o', label='cov err')
    ax_arr[0].hlines(l1_cov_err, xlims[0], xlims[1], linestyle='--',label='l1 bound')
    ax_arr[0].hlines(l2_cov_err, xlims[0], xlims[1], linestyle='-', label='l2 bound')
    ax_arr[0].set_ylabel("Covariance Reconstruction Error")
    ax_arr[0].set_xlim(xlims)
    ax_arr[0].legend(loc='best')

    # projection errors
    proj_errs = [results[t]['proj_err'] for t in changepoints]
    l1_proj_err = results['l1']['proj_err']
    l2_proj_err = results['l2']['proj_err']
    ax_arr[1].plot(changepoints, proj_errs, '-o', label='proj err')
    ax_arr[1].hlines(l1_proj_err, xlims[0], xlims[1], linestyle='--', label='l1 bound')
    ax_arr[1].hlines(l2_proj_err, xlims[0], xlims[1], linestyle='-',label='l2 bound')
    ax_arr[1].set_ylabel("Projection Error")
    ax_arr[1].set_xlim(xlims)
    ax_arr[1].legend(loc='best')

    # l hat bounds
    l1 = results['l1_size']
    l2 = results['l2_size']
    l_bounds = [results[t]['l_bound'] for t in changepoints]
    ax_arr[2].plot(changepoints, l_bounds, '-o', label='l Bound')
    ax_arr[2].hlines(l1, xlims[0], xlims[1], linestyle='--', label='l1')
    ax_arr[2].hlines(l2, xlims[0], xlims[1], linestyle='-', label='l2')
    ax_arr[2].set_ylabel("Bound on Realized L")
    ax_arr[2].set_xlabel("Changepoint")
    ax_arr[2].set_xlim(xlims)
    ax_arr[2].set_ylim(max(l1/1.1, 0), l2 * 1.1)
    ax_arr[2].legend(loc='best')

    # set up plots 
    plt.tight_layout()
    for ax in ax_arr:
        ax.grid()

    if save: 
        fig.savefig(os.path.join(results['exp_dir'], "results.png"))
    else:
        fig.show()

# fastPFD vs Batched PFD 
# maybe we should run this on (1 + \alpha)*l size?? 
def plot_tweak_batched_experiment(results_fname, save=True):
    # plot alphas vs runtime, cov_err, and reconstruction error 
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    alphas = np.sort(filter(lambda x: isinstance(x, (float)), results['tweak'].keys()))
    print "Alphas ", alphas
    xlims = [0, alphas[-1] + 0.1]
    fig, ax_arr = plt.subplots(3, sharex=True, figsize=FIG_SIZE)
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
    ax_arr[2].set_xlabel("Alpha")
    plt.tight_layout()
    for ax in ax_arr:
        ax.grid()
    for ax in ax_arr[:2]:
        ax.set_yscale('log')

    # save plot 
    if save:
        fig.savefig(os.path.join(results['exp_dir'], "results.png"))
    else:
        fig.show()

def plot_error_bpfd():
    alphas = np.arange(0.1, 1.1, 0.1)
    print "Alphas ", alphas
    xlims = [0, alphas[-1] + 0.1]
    fig, ax_arr = plt.subplots(2, sharex=True, figsize=(7, 10))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
    # from 
    batch_cov_errs = [0.000175, 0.000181, 0.000186, 0.000188, 0.000191, 0.000194, 0.000196, 0.000197, 0.000199, 0.000210]
    batch_proj_errs = [1.415417, 1.436476, 1.454169, 1.469601, 1.483738, 1.496000, 1.507494, 1.517669, 1.527717, 1.536952]
    ax_arr[0].plot(alphas, batch_cov_errs, '-o', label='BPFD Cov Err')
    ax_arr[1].plot(alphas, batch_proj_errs, '-o', label='BPFD Proj Err')
    ax_arr[0].set_ylabel("Covariance Error")
    ax_arr[1].set_ylabel("Projection Error")
    ax_arr[1].set_xlabel("Alpha")
    plt.tight_layout()
    for ax in ax_arr:
        ax.grid()
    fig.savefig('experiments/bpfd_err.png')

def plot_rand_scale():
    with open("experiments/rand_scale/results2.p", "rb") as f:
        results = pickle.load(f)
    sketch_sizes = np.sort(results.keys())
    fig = plt.figure()
    times = [results[l] for l in sketch_sizes]
    plt.plot(sketch_sizes, times, '-o')
    plt.xlabel("Sketch Size")
    plt.ylabel("Runtime (s)")
    plt.grid()
    fig.savefig('experiments/rand_scale/rand_scale2.png')

# do we also want to create a function to plot alpha experiment? 
# we have the data here on alpha choice tho! cause we also plot errors 
def plot_batched_sketch_experiment(results_fname, save=True):
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    batch_sizes = np.sort(results['rand'].keys())
    xlims = [0, batch_sizes[-1]]
    fig, ax_arr = plt.subplots(3, sharex=True, figsize=FIG_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
    # plot runtimes
    svd_times = [results['svd'][b]['time'] for b in batch_sizes]
    rand_times = [results['rand'][b]['time'] for b in batch_sizes]
    ax_arr[0].plot(batch_sizes, svd_times, '-o', label='BPFD')
    ax_arr[0].plot(batch_sizes, rand_times, '-x', label='RandBPFD')
    ax_arr[0].set_ylabel("Runtime (s)")
    ax_arr[0].legend(loc='best')
    # plot covariance errors
    svd_cov_errs = [results['svd'][b]['err'] for b in batch_sizes]
    rand_cov_errs = [results['rand'][b]['err'] for b in batch_sizes]
    ax_arr[1].plot(batch_sizes, svd_cov_errs, '-o', label='BPFD')
    ax_arr[1].plot(batch_sizes, rand_cov_errs, '-x', label='RandBPFD')
    ax_arr[1].set_ylabel("Covariance Reconstruction Error")
    ax_arr[1].legend(loc='best')
    # plot projection errors
    svd_proj_errs = [results['svd'][b]['proj_err'] for b in batch_sizes]
    rand_proj_errs = [results['rand'][b]['proj_err'] for b in batch_sizes]
    ax_arr[2].plot(batch_sizes, svd_proj_errs, '-o', label='BPFD')
    ax_arr[2].plot(batch_sizes, rand_proj_errs, '-x', label='RandBPFD')
    ax_arr[2].set_ylabel("Projection Error")
    ax_arr[2].legend(loc='best')
    ax_arr[2].set_xlabel("Batch Size")
    #ax_arr[2].set_scale("log")

    # format plot 
    plt.tight_layout()
    for ax in ax_arr:
        ax.grid()
        #ax.set_yscale('log')
        ax.set_xlim(xlims)
    if save:
        fig.savefig(os.path.join(results['exp_dir'], "results.png"))
    else:
        fig.show()

def plot_parallel_sketch_experiment(results_fname, save=True):
	with open(results_fname, "rb") as f:
	    results = pickle.load(f)
	processors = np.sort(results['rand'].keys())
	xlims = [0, processors[-1]]
	fig, ax_arr = plt.subplots(3, sharex=True, figsize=FIG_SIZE)
	plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))

	svd_times = [results['svd'][p]['time'] for p in processors]
	rand_times = [results['rand'][p]['time'] for p in processors]
	ax_arr[0].plot(processors, svd_times, '-o', label='Deterministic SVD Runtime')
	ax_arr[0].plot(processors, rand_times, '-x', label='Randomized SVD Runtime')
	ax_arr[0].set_ylabel("Runtime (s)")
	ax_arr[0].legend(loc='best')

	svd_cov_errs = [results['svd'][p]['err'] for p in processors]
	rand_cov_errs = [results['rand'][p]['err'] for p in processors]
	ax_arr[1].plot(processors, svd_cov_errs, '-o', label='Deterministic SVD Cov Err')
	ax_arr[1].plot(processors, rand_cov_errs, '-x', label='Randomized SVD Cov Err')
	ax_arr[1].set_ylabel("Covariance Reconstruction Error")
	ax_arr[1].legend(loc='best')
	# plot projection errors
	svd_proj_errs = [results['svd'][p]['proj_err'] for p in processors]
	rand_proj_errs = [results['rand'][p]['proj_err'] for p in processors]
	ax_arr[2].plot(processors, svd_proj_errs, '-o', label='Deterministic SVD Proj Err')
	ax_arr[2].plot(processors, rand_proj_errs, '-x', label='Randomized SVD Proj Err')
	ax_arr[2].set_ylabel("Projection Error")
	ax_arr[2].legend(loc='best')
	ax_arr[2].set_xlabel("Cores")

	# format plot 
	plt.tight_layout()
	for ax in ax_arr:
	    ax.grid()
	    ax.set_xlim(xlims)
	if save:
	    fig.savefig(os.path.join(results['exp_dir'], "results.png"))
	else:
	    fig.show()

sketch_t_to_name = {'jlt': 'JLT', 'cw': 'CW', 'batch-pfd': 'BPFD', 'fd': 'FD', 'pfd': 'PFD'}
def plot_comparison_experiment(results_fname, save=True):
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    sketch_sizes = np.sort(results['jlt'].keys())
    # now what do I do? let's make 
    xlims = [0, sketch_sizes[-1] * 1.1]
    fig, ax_arr = plt.subplots(3, sharex=True, figsize=FIG_SIZE)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
    all_sketches = ['batch-pfd', 'pfd', 'fd', 'jlt', 'cw']
    
    for sketch_t in all_sketches:
        times = [results[sketch_t][l]['time'] for l in sketch_sizes]
        errs = [results[sketch_t][l]['err'] for l in sketch_sizes]
        proj_errs = [results[sketch_t][l]['proj_err'] for l in sketch_sizes]
        # plot 
        ax_arr[0].plot(sketch_sizes, times, '-o', label=sketch_t_to_name[sketch_t])
        ax_arr[1].plot(sketch_sizes, errs, '-o', label=sketch_t_to_name[sketch_t])
        ax_arr[2].plot(sketch_sizes, proj_errs, '-o', label=sketch_t_to_name[sketch_t])
    
    ax_arr[0].set_ylabel("Runtime (s)")
    ax_arr[1].set_ylabel("Covariance Error")
    ax_arr[2].set_ylabel("Projection Error")
    ax_arr[2].set_xlabel("Sketch Size")
    for ax in ax_arr:
        ax.grid()
        ax.set_yscale("log")
        ax.legend(loc='best')
    plt.tight_layout()
    if save:
        exp_dir = os.path.split(results_fname)[0]
        fig.savefig(os.path.join(exp_dir, "results.png"))
    else:
        fig.show()

def plot_scalability(results_fname="experiments/parallel_ESOC/ESOC/results.p"):
    with open(results_fname, "rb") as f:
        results = pickle.load(f)
    fig = plt.figure()
    cores = [2, 4, 8, 16]
    svd_data = []
    rand_data = []
    svd_times =results['svd']
    rand_times = results['rand']
    print svd_times
    svd_norm = svd_times[2]
    rand_norm = rand_times[2]
    for c in cores:
        if c in svd_times:
            svd_data.append((c, np.array(svd_norm / svd_times[c]) ))
        if c in rand_times:
            rand_data.append((c, np.array(float(rand_norm) / rand_times[c]) ))
    svd_cores, svd_times = zip(*svd_data)
    rand_cores, rand_times = zip(*rand_data)
    #plt.xscale('log')
    plt.plot(svd_cores, svd_times, "-o", label="svd")
    plt.plot(rand_cores, rand_times, "-o", label="rand")
    plt.plot(cores, np.array(cores)/2, '--', label='opt')
    plt.xlabel("Cores")
    plt.ylabel("Speedup")
    plt.grid()
    plt.legend(loc='best')
    fig_path = os.path.split(results_fname)[0]
    fig.savefig(os.path.join(fig_path, "results.png"))
    plt.show()


def plot_kmeans(sketch_results_fname='experiments/kmeans/cifar/sketch_results.p', mat_results_fname='experiments/kmeans/cifar/mat_results.p'):
    # what does the plot look like
    plot_large = False
    fig = plt.figure()
    results = {}
    with open(sketch_results_fname, "rb") as f:
        results['sketch'] = pickle.load(f)['sketch']
    with open(mat_results_fname, "rb") as f:
        results['opt'] = pickle.load(f)['opt']
    if plot_large:
        with open("experiments/kmeans/cifar/large_sketch_results.p") as f:
            large_results = pickle.load(f)['sketch']
            print large_results.keys()
    clusters = [5, 10, 15, 20]
    opt_data = []
    sketch_data = []
    large_sketch_data = []
    for k in clusters:
        opt_data.append((results['opt'][k]['time'], results['opt'][k]['cost']))
        sketch_data.append((results['sketch'][k]['time'], results['sketch'][k]['cost']))
        if plot_large:
            large_sketch_data.append((large_results[k]['time'], large_results[k]['cost']))
    opt_times, opt_costs = zip(*opt_data)
    sketch_times, sketch_costs = zip(*sketch_data)
    if plot_large:
        large_sketch_times, large_sketch_costs = zip(*large_sketch_data)
    print opt_times 
    print np.array(sketch_times) + 64
    print sketch_costs
    print opt_costs
    plt.plot(clusters, np.array(sketch_costs)/np.array(opt_costs), '-o', label = 'l=200')
    if plot_large:
        plt.plot(clusters, np.array(large_sketch_costs)/np.array(opt_costs), '-o', label = 'l=1000')
    #plt.plot(clusters, np.array(sketch_costs), '-o', label='sketch')
    #plt.yscale('log')
    #plt.plot(clusters, opt_costs, '-o', label="opt")
    # what do we do for times? 
    plt.xlabel("Clusters")
    plt.ylabel("K-Means Approximation Error (OPT/Sketch)")
    plt.legend(loc='best')
    fig_path = os.path.split(sketch_results_fname)[0]
    #plt.ylim((6, 9))
    plt.grid()
    fig.savefig(os.path.join(fig_path, "results.png"))


# SAMPLE FILENAMES
#fname = "experiments/tweak_batch_exp_small_data_batch_1/small_data_batch_1/results.p"
#fname = "experiments/dynamic_exp_cifar_data_200_600/cifar_data/results.p"
#fname = "experiments/rand_batch_exp_cifar_data/cifar_data/results.p"
#fname = "experiments/tweak_batch_exp_data_batch_1/data_batch_1/results.p"
#mat_fname = 'data_batch_1'
#path = "experiments/tweak_batch_exp_%s/%s/results.p" %(mat_fname, mat_fname)
#fname = "experiments/rand_batch_exp_cifar_data/cifar_data/results.p"
#fname = "experiments/parallel_exp_cifar_data/cifar_data/results.p"

if __name__ == "__main__":
    #plot_scalability()
    #path = "experiments/dynamic_exp_cifar_data_200_600/cifar_data/results.p"
    #path = "experiments/tweak_batch_exp_data_batch_1/data_batch_1/results.p"
    #plot_tweak_batched_experiment(path)
    plot_kmeans()
    #plot_comparison_experiment("experiments/sketch_exp_data_batch_1/data_batch_1/results.p")
    #plot_dynamic_sketch_experiment("experiments/dynamic_exp_cifar_data_200_600/cifar_data/results.p")
    #plot_batched_sketch_experiment("experiments/rand_batch_exp_cifar_data/cifar_data/results.p")
    #plot_rand_scale()
    #plot_lbounds_err("experiments/dynamic_exp_cifar_data_200_600/cifar_data/results.p")
