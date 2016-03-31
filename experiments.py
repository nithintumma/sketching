import os 
import time 
import re 
import subprocess
import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt 
import cPickle as pickle 
from gensim import models
from helpers import load_matrix, write_matrix
from fd_sketch import (JLTSketch, CWSparseSketch, FDSketch, BatchFDSketch, PFDSketch, 
                        BatchPFDSketch, DynamicFDSketch, TweakPFDSketch, calculateError, 
                        sparse_calculate_error, calculate_projection_error, 
                        squaredFrobeniusNorm) 
from parallel_sketch import parallel_bpfd_sketch, sparse_parallel_bpfd_sketch

# CONSTANT DIRECTORIES  
MATRIX_DIR = 'test_matrices'
RUN_SKETCH = './sketch'
EXP_DIR = './experiments/'

# experiment class where we are plotting one sketch, time and error 
class Experiment(object):
    """
    how do we run experiments efficiently and store their results? 
    """
    def __init__(self, exp_name, mat_fname, dependent_vars, dependent_var_name, sparse=False):
        self.exp_name = exp_name
        self.mat_fname = mat_fname
        if sparse:
            # assume Matrix Market format
            self.mat = mmread(os.path.join(MATRIX_DIR, mat_fname))
        else:
            self.mat = load_matrix(self.mat_fname)
        self.exp_dir = os.path.join(EXP_DIR, exp_name, os.path.splitext(mat_fname)[0])
        # make a directory for the experiment if it doesnt exist yet 
        try:
            os.makedirs(self.exp_dir)
        except OSError, e:
            if e.errno != 17:
                raise
            pass
        if hasattr(self, "results"):
            self.results['exp_dir'] = self.exp_dir
            self.results['mat_fname'] = self.mat_fname
        else:
            self.results = {"exp_dir": self.exp_dir, 
                            "mat_fname": self.mat_fname}
        self.computed_results = False
        self.dependent_vars = dependent_vars
        self.dependent_var_name = dependent_var_name

    def _set_plot(self):
        # use this to configure plots
        plt.grid()

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        if (self.dependent_vars is None) or (self.results is None):
            raise Exception("Need to set dependent vars and run experiment")
        if err:
            # plot error as a function of batch size 
            fig = plt.figure()
            self._set_plot()
            errs = [self.results[d]["err"] for d in self.dependent_vars]
            plt.plot(self.dependent_vars, errs, '-o', color='b', label='cov err')
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Covariance Reconstruction Error")
            if save:
                fig.savefig(os.path.join(self.exp_dir, "err_plt.png"))
            else:
                fig.show()
        if proj_err:
            fig = plt.figure()
            self._set_plot()
            errs = [self.results[d]["proj_err"] for d in self.dependent_vars]
            plt.plot(self.dependent_vars, errs, '-o', color='b', label='cov err')
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Projection Error")
            if save:
                fig.savefig(os.path.join(self.exp_dir, "proj_err_plt.png"))
            else:
                fig.show()
        if time: 
            fig = plt.figure()
            self._set_plot()
            times = [self.results[d]["time"] for d in self.dependent_vars]
            plt.plot(self.dependent_vars, times, '-o', color='b', label='runtime')
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Runtime (s)")
            if save:
                fig.savefig(os.path.join(self.exp_dir, "time_plt.png"))
            else:
                fig.show()

    def write_results(self, header=" ", only_pickle=False):
        if self.results is None:
            raise Exception("Need to compute results")
        if not only_pickle:
            with open(os.path.join(self.exp_dir, "results.txt"), "wb") as r_file:
                r_file.write("%s\n"%header)
                for d in self.dependent_vars:
                    rtime, err, proj_err = self.results[d]['time'], self.results[d]['err'], self.results[d]['proj_err']
                    r_file.write("%s: %d, Runtime: %f secs, Err: %f, Proj Err: %f\n" %(self.dependent_var_name, d, rtime, err, proj_err))
        # pickle results 
        with open(os.path.join(self.exp_dir, "results.p"), "wb") as p_file:
            pickle.dump(self.results, p_file)

class DynamicSketchExperiment(Experiment):
    """
    run test with fixed l1, l2, and varying change points 
    """
    def __init__(self, exp_name, mat_fname, l1, l2, batch_size, change_points, randomized=False):
        self.l1 = l1
        self.l2 = l2
        # so we can run multiple ones at the same time 
        exp_name = "%s_%d_%d" %(exp_name, l1, l2)
        self.change_points = change_points
        self.randomized = randomized
        self.batch_size = batch_size
        self.results = {}
        self.results['l1_size'] = l1
        self.results['l2_size'] = l2
        self.results['batch_size'] = batch_size
        assert(self.l1 + self.batch_size > self.l2)
        super(DynamicSketchExperiment, self).__init__(exp_name, mat_fname, change_points, "Change Point")

    def run_experiment(self):
        # compute sketches for each batch size, then save the results to a dictionary, 
        sketch_objs = []
        for t in self.change_points:
            print "Testing: ", t
            sketch_obj = DynamicFDSketch(self.mat, self.l1, self.l2, t, self.batch_size, randomized=self.randomized)
            # compute the sketch 
            sketch_obj.compute_sketch()
            self.results[t] = {"time": sketch_obj.sketching_time, 
                                "err": sketch_obj.sketch_err(),
                                "proj_err": sketch_obj.sketch_projection_err(k=100),
                                "l_bound": sketch_obj.compute_actual_l_bound()}
            sketch_objs.append(sketch_obj)
        #l1_sketch = BatchFDSketch(self.mat, self.l1, self.batch_size + self.l2 - self.l1)
        l1_sketch = BatchFDSketch(self.mat, self.l1, self.batch_size)
        l1_sketch.compute_sketch()
        self.results["l1"] = {"time": l1_sketch.sketching_time, 
                              "err": l1_sketch.sketch_err(),
                              "proj_err": l1_sketch.sketch_projection_err(k=100)}
        sketch_objs.append(l1_sketch)
        l2_sketch = BatchFDSketch(self.mat, self.l2, self.batch_size)
        l2_sketch.compute_sketch()
        self.results["l2"] = {"time": l2_sketch.sketching_time, 
                              "err": l2_sketch.sketch_err(),
                              "proj_err": l2_sketch.sketch_projection_err(k=100)}
        sketch_objs.append(l2_sketch)
        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results

    def write_results(self):
        # we don't write the l1, l2 results 
        if not self.computed_results:
            self.run_experiment()
        if self.randomized:  
            title = "Randomized Dynamic FD Sketch Experiment"
        else:
            title = "Dynamic FD Sketch Experiment"			
        header = "%s: %s, Matrix: %s, Initial Sketch Size: %d, End Sketch Size: %d, Batch Size: %d\n" %(title, self.exp_name, self.mat_fname, self.l1, self.l2, self.batch_size)
        super(DynamicSketchExperiment, self).write_results(header)

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        raise Exception("Not supported ATM")
        super(DynamicSketchExperiment, self).plot_results(err=False, proj_err=False, time=time, save=save)

class AlphaSketchExperiment(Experiment):
    """
    run experiment on changing alpha  
    want to check runtime of C code, so need to call with subprocess and time it 
    """
    def __init__(self, exp_name, mat_fname, l, alphas = np.arange(0.1, 1.1, 0.1)):
        self.alphas = alphas
        self.l = l
        super(AlphaSketchExperiment, self).__init__(exp_name, mat_fname, alphas, "Alpha")

    def run_experiment(self):
        sketch_objs = []
        for a in self.alphas:
            print "Testing: ", a
            sketch_obj = PFDSketch(self.mat, self.l, a)
            sketch_obj.compute_sketch()
            self.results[a] = {"time": sketch_obj.sketching_time, 
                                "err": sketch_obj.sketch_err(),
                                "proj_err": sketch_obj.sketch_projection_err()}
            sketch_objs.append(sketch_obj)
        self.sketch_objs = sketch_objs
        self.computed_results =  True
        return self.results

    def write_results(self):
        if not self.computed_results:
            self.run_experiment()
        title = "Alpha PFD Experiment"
        header = "%s: %s, Matrix: %s, Sketch Size: %d\n" %(title, self.exp_name, self.mat_fname, self.l)
        super(AlphaSketchExperiment, self).write_results(header)

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        if not self.computed_results:
            self.run_experiment()
        super(AlphaSketchExperiment, self).plot_results(err, proj_err, time, save)

SUPPORTED_SKETCHES = set(['jlt', 'cw', 'fd', 'pfd', 
                            'batch-pfd','rand-batch-pfd'])
class SketchExperiment(Experiment):
    def __init__(self, exp_name, mat_fname, sketch_sizes, sketch_types=list(SUPPORTED_SKETCHES)):
        """
        sketch_types should be a dict, keys are SUPPORTED SKETCHES and vals are params for them 
        """
        self.sketches ={}
        for sketch_t in sketch_types.keys():
            assert(sketch_t in SUPPORTED_SKETCHES) 
        self.sketch_types = sketch_types
        self.sketch_sizes = sketch_sizes
        super(SketchExperiment, self).__init__(exp_name, mat_fname, sketch_sizes, "Sketch Size")


    def run_experiment(self):
        # for each sketch, for sketch size, create an approprite sketch object
        sketch_objs = []
        for sketch_t in self.sketch_types.keys():
            print "Testing ", sketch_t
            _sketch_class_args = []
            _sketch_class_kwargs = {}
            if sketch_t == 'jlt':
                _sketch_class = JLTSketch
            elif sketch_t == 'cw':
                _sketch_class = CWSparseSketch
            elif sketch_t == 'fd':
                _sketch_class = FDSketch
            elif sketch_t == 'pfd':
                _sketch_class = PFDSketch
                _sketch_class_args = [self.sketch_types[sketch_t]['alpha']]
            elif sketch_t == 'batch-pfd':
                _sketch_class = BatchPFDSketch
                # batch_size and lpah
                _sketch_class_args = [self.sketch_types[sketch_t]['batch_size'], 
                                        self.sketch_types[sketch_t]['alpha']]
            elif sketch_t == 'rand-batch-pfd':
                _sketch_class = BatchPFDSketch
                _sketch_class_args = [self.sketch_types[sketch_t]['batch_size'], 
                                        self.sketch_types[sketch_t]['alpha']]
                _sketch_class_kwargs['randomized'] = True
            # now compute the sketches for each of the sketch sizes 
            self.results[sketch_t] = {}
            for l in self.sketch_sizes: 
                sketch_obj = _sketch_class(self.mat, l, 
                                *_sketch_class_args, **_sketch_class_kwargs)
                sketch_obj.compute_sketch()
                self.results[sketch_t][l] = {"time": sketch_obj.sketching_time, 
                                             "err": sketch_obj.sketch_err(),
                                             "proj_err": sketch_obj.sketch_projection_err()}
                sketch_objs.append(sketch_obj)

        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results

    def write_results(self):
        super(SketchExperiment, self).write_results(only_pickle=True)

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        if not self.computed_results:
            self.computed_results()
        if err:
            fig = plt.figure()
            self._set_plot()
            for sketch_t in self.results.keys():
                errs = [self.results[sketch_t][l]['err'] for l in self.sketch_sizes]
                plt.plot(self.sketch_sizes, errs, '-o', label=sketch_t)
            plt.legend(loc="best")
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Covariance Reconstruction Error")
            plt.yscale('log')
            if save:
                fig.savefig(os.path.join(self.exp_dir, "err_plt.png"))
            else:
                fig.show()
        if proj_err: 
            fig = plt.figure()
            self._set_plot()
            for sketch_t in self.results.keys():
                errs = [self.results[sketch_t][l]['proj_err'] for l in self.sketch_sizes]
                plt.plot(self.sketch_sizes, errs, '-o', label=sketch_t)
            plt.legend(loc="best")
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Projection Error")
            plt.yscale('log')
            if save:
                fig.savefig(os.path.join(self.exp_dir, "proj_err_plt.png"))
            else:
                fig.show()
        if time: 
            fig = plt.figure()
            self._set_plot()
            for sketch_t in self.results.keys():
                times = [self.results[sketch_t][l]['time'] for l in self.sketch_sizes]
                plt.plot(self.sketch_sizes, times, '-o', label=sketch_t)
            plt.legend(loc="best")
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Sketching Time (s)")
            plt.yscale('log')
            if save:
                fig.savefig(os.path.join(self.exp_dir, "time_plt.png"))
            else:
                fig.show()

class TweakVsBatchPFDSketchExperiment(Experiment):
    # we should run tweaked on 2l rows? then it is comparable to our sketch on l rows? 
    """
    compare tweaked PFD with batched PFD for a range of alphas 
    goal is to show that our runtime is invariant to alpha, but theres is not 
    runs is number of times that we should repeat the experimeent for runtime 
    """
    def __init__(self, exp_name, mat_fname, l, alphas, runs=3, 
                    randomized=False, fast=True, double=True):
        self.l = l
        self.alphas = alphas
        self.runs = runs
        self.randomized = randomized
        self.fast = fast
        self.double=double
        print "Fast: ", self.fast
        super(TweakVsBatchPFDSketchExperiment, self).__init__(exp_name, 
                                            mat_fname, alphas, "Alpha")

    def run_experiment(self):
        sketch_objs = []
        self.results['tweak'] = {}
        self.results['batch'] = {}
        for a in self.alphas:
            print "Testing ", a
            times = []
            for i in range(self.runs):
                sketch_obj = BatchPFDSketch(self.mat, self.l, 
                                            self.l, a, 
                                            randomized=self.randomized)
                # compute the sketch 
                sketch_obj.compute_sketch()
                times.append(sketch_obj.sketching_time)
            self.results["batch"][a] = {"time": np.mean(times), 
                                        "err": sketch_obj.sketch_err(),
                                        "proj_err": sketch_obj.sketch_projection_err(k=self.l/2)}
            sketch_objs.append(sketch_obj)
            times = []
            for i in range(self.runs):
                #TODO: do we actually want 2 l here? 
                l = self.l
                if self.double:
                    l = 2 * self.l
                sketch_obj = TweakPFDSketch(self.mat, l, a, fast=self.fast)
                sketch_obj.compute_sketch()
                times.append(sketch_obj.sketching_time)
            sketch_obj.sketch = sketch_obj.sketch[:self.l, :]
            self.results["tweak"][a] = {"time": np.mean(times), 
                                        "err": sketch_obj.sketch_err(),
                                        "proj_err": sketch_obj.sketch_projection_err(k=self.l/2)}

            sketch_objs.append(sketch_obj)
        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results

    def write_results(self):
        super(TweakVsBatchPFDSketchExperiment, self).write_results(only_pickle=True)

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        raise Exception("Not implemented")

class BatchSketchExperiment(Experiment):
    """
    determine which batch size to choose for a given l, matrix 
    """
    def __init__(self, exp_name, mat_fname, l, batch_sizes, runs=3, randomized=False):
        self.l = l
        self.batch_sizes = batch_sizes
        self.randomized = randomized
        self.runs = runs
        super(BatchSketchExperiment, self).__init__(exp_name, mat_fname, batch_sizes, "Batch Size")

    def run_experiment(self):
        # compute sketches for each batch size, then save the results to a dictionary, 
        sketch_objs = []
        for b_size in self.batch_sizes:
            sketch_obj = BatchFDSketch(self.mat, self.l, b_size, randomized=self.randomized)
            # compute the sketch 
            sketch_obj.compute_sketch()
            self.results[b_size] = {"time": sketch_obj.sketching_time, 
                                    "err": sketch_obj.sketch_err(),
                                    "proj_err": sketch_obj.sketch_projection_err()}
            sketch_objs.append(sketch_obj)
        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results

    def write_results(self):
        if not self.computed_results:
            self.run_experiment()
        if self.randomized:  
            title = "Randomized Batched FD Sketch Experiment"
        else:
            title = "Batched FD Sketch Experiment"
        header = "%s: %s, Matrix: %s, Sketch Size: %d\n" %(title, self.exp_name, self.mat_fname, self.l)
        super(BatchSketchExperiment, self).write_results(header)

    def plot_results(self, err=True, proj_err=True, time=True, save=True):
        # what do we need to do here? lets plot the time and error as a function of batch_size 
        if not self.computed_results:
            self.run_experiment()
        super(BatchSketchExperiment, self).plot_results(err=False, proj_err=proj_err, time=time, save=save)

class BatchRandomPFDSketchExperiment(Experiment):
    """
    leave l fixed
    leave alpha fixed
    change batch size
    compare runtime of Batch PFD with Randomized Batched PFD
    """
    def __init__(self, exp_name, mat_fname, l, alpha, batch_sizes, runs=1):
        # should we update the exp_name to include alpha? probably 
        self.l = l
        self.alpha = alpha
        self.batch_sizes = batch_sizes
        # number of trials to average over for timing, err 
        self.runs = runs
        super(BatchRandomPFDSketchExperiment, self).__init__(exp_name, mat_fname, batch_sizes, "Batch Size")

    def run_experiment(self):
        self.results['rand'] = {}
        self.results['svd'] = {}
        sketch_objs = []
        for b in self.batch_sizes:
            print "Testing: ", b
            svd_results = []
            rand_results = []
            for i in range(self.runs):
                svd_sketch = BatchPFDSketch(self.mat, self.l, b, self.alpha, randomized=False)
                svd_sketch.compute_sketch()
                svd_results.append((svd_sketch.sketching_time, 
                                    svd_sketch.sketch_err(), 
                                    svd_sketch.sketch_projection_err()))

                rand_sketch = BatchPFDSketch(self.mat, self.l, b, self.alpha, randomized=True)
                rand_sketch.compute_sketch()
                rand_results.append((rand_sketch.sketching_time, 
                                    rand_sketch.sketch_err(), 
                                    rand_sketch.sketch_projection_err()))

            svd_times, svd_errs, svd_proj_errs = zip(*svd_results)
            self.results['svd'][b] = {'time': np.mean(svd_times),
                                        'err': np.mean(svd_errs),
                                        'proj_err': np.mean(svd_proj_errs)}

            rand_times, rand_errs, rand_proj_errs = zip(*rand_results)            
            self.results['rand'][b] = {"time": np.mean(rand_times),
                                        'err': np.mean(rand_errs), 
                                        'proj_err': np.mean(rand_proj_errs)}
        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results 

    def write_results(self):
        super(BatchRandomPFDSketchExperiment, self).write_results(only_pickle=True)
 
    def plot_results(self):
        # call the appropriate function from plot results 
        raise Exception("Not implemented yet")

# we want to see how the non-randomized and randomized BPFD scale with number of cores 
class ParallelPFDSketchExperiment(Experiment):
    def __init__(self, exp_name, mat_fname, l, alpha, batch_size, processors=[1, 2, 3, 4], runs=2, sparse=False):
        self.l = l
        self.alpha = alpha
        self.batch_size = batch_size
        self.processors = processors 
        self.runs = runs
        self.sparse = sparse
        super(ParallelPFDSketchExperiment, self).__init__(exp_name, mat_fname, processors, "Cores", sparse=sparse)

    def run_experiment(self):
        if self.sparse:
            parallel_sketch_func = sparse_parallel_bpfd_sketch
        else:
            parallel_sketch_func = parallel_bpfd_sketch
        self.results['rand'] = {}
        self.results['svd'] = {} 
        sketch_objs = []
        for p in self.processors:
            print "Testing: ", p
            svd_results = []
            rand_results = []
            for i in range(self.runs):
                svd_start_time = time.time()
                svd_sketch = parallel_sketch_func(self.mat, self.l, self.alpha, self.batch_size,
                                                    randomized=False, num_processes=p)
                svd_time = time.time() - svd_start_time
                if not self.sparse:
                    svd_err = calculateError(self.mat, svd_sketch)
                    svd_proj_err = calculate_projection_error(self.mat, svd_sketch, k=100)
                else:
                    svd_err = sparse_calculate_error(self.mat, svd_sketch)
                    # not implemented yet
                    svd_proj_err = 1.0
                svd_results.append((svd_time, svd_err, svd_proj_err))
                sketch_objs.append(svd_sketch)

                rand_start_time = time.time()
                rand_sketch = parallel_sketch_func(self.mat, self.l, self.alpha, self.batch_size,
                                                    randomized=True, num_processes=p)
                rand_time = time.time() - rand_start_time
                if not self.sparse:
                    rand_err = calculateError(self.mat, rand_sketch)
                    rand_proj_err = calculate_projection_error(self.mat, rand_sketch, k=100)
                else:
                    rand_err = sparse_calculate_error(self.mat, rand_sketch)
                    # not implemented yet 
                    rand_proj_err = 1.0
                rand_results.append((rand_time, rand_err, rand_proj_err))
                sketch_objs.append(rand_sketch)

            svd_times, svd_errs, svd_proj_errs = zip(*svd_results)
            self.results['svd'][p] = {'time': np.mean(svd_times),
                                        'err': np.mean(svd_errs),
                                        'proj_err': np.mean(svd_proj_errs)}

            rand_times, rand_errs, rand_proj_errs = zip(*rand_results)
            self.results['rand'][p] = {'time': np.mean(rand_times),
                                        'err': np.mean(rand_errs),
                                        'proj_err': np.mean(rand_proj_errs)}

        self.sketch_objs = sketch_objs
        self.computed_results = True
        return self.results 

    def write_results(self):
        super(ParallelPFDSketchExperiment, self).write_results(only_pickle=True)

    def plot_results(self):
        raise Exception("Not Implemented")

from cluster import train_kmeans, kmeans_objective, compute_cost_labels
def kmeans_experiment(on_sketch=True, on_orig=True):
    #path = "../../data/GoogleNews-vectors-negative300.bin"
    #wmodel = models.Word2Vec.load_word2vec_format(path, binary=True)
    mat = load_matrix('data_batch_1')
    # sketch the transpose 
    mat = mat.T
    sketch_sizes = [50, 100, 200]
    sketch_objs = [BatchPFDSketch(mat, l, l, 0.2, randomized=True) for l in sketch_sizes]
    sketches = []
    for sk in sketch_objs:
        sketches.append(sk.compute_sketch().T)
    mat = mat.T
    #print sketche.shape
    print "Mat: ", mat.shape
    #sketch = load_matrix("sketches/w2vec_250.txt")
    clusters = [5, 10, 15, 20]
    num_processes = 8
    results = {'opt': {}, 'sketch': {}}
    for l in sketch_sizes:
        results['sketch'][l] = {}

    for k in clusters:
        print "Testing ", k
        if on_sketch:
            for sketch, l in zip(sketches, sketch_sizes):
                start_time = time.time()
                cost, cluster_centers, labels = train_kmeans(sketch, k, 
                                                    num_processes=num_processes)
                train_time = time.time() - start_time
                test_cost = compute_cost_labels(mat, labels, k)
                results['sketch'][l][k] = {'time': train_time, 'cost': test_cost}
        if on_orig:
            start_time = time.time()
            cost, cluster_centers, labels = train_kmeans(mat, k, num_processes=num_processes)
            train_time = time.time() - start_time
            results['opt'][k] = {'time': train_time, 'cost': cost}
    
    if on_orig:
        with open('experiments/kmeans/cifar/mat_results.p', "wb") as f:
            pickle.dump(results, f)
    if on_sketch:
        with open('experiments/kmeans/cifar/sketch_results.p', "wb") as f:
            pickle.dump(results, f)


def test_batch_exp():
    mat_fname = "med_svd_mat.txt"
    l = 30
    batch_sizes = np.arange(1, 2*l, max(l/5, 1))
    exp_name = "test_batch_exp"
    svd_batch_exp = BatchSketchExperiment(exp_name, mat_fname,
                                            l, batch_sizes,
                                            randomized=False)
    rand_batch_exp = BatchSketchExperiment(exp_name, mat_fname,
                                            l, batch_sizes,
                                            randomized=True)
    svd_batch_exp.run_experiment()
    rand_batch_exp.run_experiment()
    plot_batched_experiments(svd_batch_exp, rand_batch_exp)

def test_alpha_exp():
    mat_fname = "med_svd_mat.txt"
    l = 30
    alphas = np.arange(0.1, 1, 0.1)
    print alphas
    exp_name = 'test_pfd_exp'
    pfd_exp = AlphaSketchExperiment(exp_name, mat_fname, l, alphas)
    pfd_exp.run_experiment()
    pfd_exp.write_results()
    pfd_exp.plot_results(err=True, time=True, save=True)

def test_sketch_exp():
    mat_fname = "med_svd_mat.txt"
    ls = np.arange(10, 40, 5)
    exp_name = "test_sketch_exp"
    sketch_types = {'jlt': None, 'cw':None, 'fd': None, 'pfd': {'alpha': 0.2}, 'batch-pfd': {'batch_size': 30, 'alpha': 0.2}}
    sketch_exp = SketchExperiment(exp_name, mat_fname, ls, sketch_types=sketch_types)
    sketch_exp.run_experiment()
    sketch_exp.write_results()
    sketch_exp.plot_results(err=True, time=True, save=True)

def test_dyn_exp():
    mat_fname = "med_svd_mat.txt"
    l1 = 20
    l2 = 30
    batch_size = 20
    mat = load_matrix(mat_fname)
    ts = np.arange(0, mat.shape[0], max(mat.shape[0]/10, 1))
    exp_name = "test_dynamic_exp"
    dyn_exp = DynamicSketchExperiment(exp_name, mat_fname, l1, l2, batch_size, ts, randomized=False)
    dyn_exp.run_experiment()
    dyn_exp.plot_results()
    dyn_exp.write_results()

def test_rand_exp():
    mat_fname = "small_data_batch_1"
    l = 200
    sketch_sizes = [100, 200, 300, 400, 500]
    batch_sizes =[5, 10, 20]
    alpha = 0.2
    exp_name = 'test_rand_bpfd_experiment'
    mat = load_matrix(mat_fname)
    print mat.shape
    results = {}
    for l in sketch_sizes:
        sk_o = BatchPFDSketch(mat, l, 5000, alpha, randomized=True)
        sk_o.compute_sketch()
        results[l] = sk_o.sketching_time
    with open("experiments/rand_scale/results2.p", "wb") as f:
        pickle.dump(results, f) 
    print "Done"
    #exp = BatchRandomPFDSketchExperiment(exp_name, mat_fname, l, alpha, batch_sizes, runs=3)
    #exp.run_experiment()
    #exp.write_results()

def test_par_exp():
    mat_fname = "med_svd_mat.txt"
    l = 10
    batch_size = 10
    processors =[1, 2, 4]
    alpha = 0.2
    exp_name = "test_par_exp"
    exp = ParallelPFDSketchExperiment(exp_name, mat_fname, l, alpha, batch_size, processors=processors, runs=2)
    exp.run_experiment()
    exp.write_results()


if __name__ == "__main__":
    #kmeans_experiment(on_sketch=True, on_orig=True)
    #test_rand_exp()
    kmeans_experiment(on_sketch=True, on_orig=False)
    #test_rand_exp()
