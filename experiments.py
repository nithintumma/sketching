import os 
import subprocess
import numpy as np
import time 
import re 
import matplotlib.pyplot as plt 
import cPickle as pickle 


from helpers import load_matrix, write_matrix
from fd_sketch import (JLTSketch, CWSparseSketch, FDSketch, BatchFDSketch, PFDSketch, 
        BatchPFDSketch, DynamicFDSketch, calculateError, squaredFrobeniusNorm) 

# CONSTANT DIRECTORIES  
MATRIX_DIR = 'test_matrices'
RUN_SKETCH = './sketch'
EXP_DIR = './experiments/'

# experiment class where we are plotting one sketch, time and error 
class Experiment(object):
    """
    how do we run experiments efficiently and store their results? 
    """
    def __init__(self, exp_name, mat_fname, dependent_vars, dependent_var_name):
        self.exp_name = exp_name
        self.mat_fname = mat_fname
        self.mat = load_matrix(self.mat_fname)
        self.exp_dir = os.path.join(EXP_DIR, exp_name, os.path.splitext(mat_fname)[0])
        # make a directory for the experiment if it doesnt exist yet 
        try:
            os.makedirs(self.exp_dir)
        except OSError, e:
            if e.errno != 17:
                raise
            pass
        if hasattr(self, results):
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
        self.change_points = change_points
        self.randomized = randomized
        self.batch_size = batch_size
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
                                "proj_err": sketch_obj.sketch_projection_err(),
                                "l_bound": sketch_obj.compute_actual_l_bound()}
            sketch_objs.append(sketch_obj)
        # add l1, l2 full sketches 
        # is this necessary? 
        #l1_sketch = BatchFDSketch(self.mat, self.l1, self.batch_size + self.l2 - self.l1)
        l1_sketch = BatchFDSketch(self.mat, self.l1, self.batch_size)
        l1_sketch.compute_sketch()
        self.results["l1"] = {"time": l1_sketch.sketching_time, 
                              "err": l1_sketch.sketch_err(),
                              "proj_err": l1_sketch.sketch_projection_err()}
        sketch_objs.append(l1_sketch)
        l2_sketch = BatchFDSketch(self.mat, self.l2, self.batch_size)
        l2_sketch.compute_sketch()
        self.results["l2"] = {"time": l2_sketch.sketching_time, 
                              "err": l2_sketch.sketch_err(),
                              "proj_err": l2_sketch.sketch_projection_err()}
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
        if not self.computed_results:
            self.run_experiment()
        # we need to add the l1 and l2 sketch errs to the error plot 
        if (self.dependent_vars is None) or (self.results is None):
            raise Exception("Need to set dependent vars and run experiment")
        if err:
            # plot error as a function of batch size 
            fig = plt.figure()
            self._set_plot()
            errs = [self.results[d]["err"] for d in self.dependent_vars]
            plt.plot(self.dependent_vars, errs, '-o', color='b', label='cov err')
            l1_err = self.results['l1']['err']
            l2_err = self.results['l2']['err']
            print self.dependent_vars[0], self.dependent_vars[-1]
            plt.hlines(l1_err, self.dependent_vars[0], self.dependent_vars[-1], "g", label='l1 Err')
            plt.hlines(l2_err, self.dependent_vars[0], self.dependent_vars[-1], "r", label='l2 Err')
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
            plt.plot(self.dependent_vars, errs, '-o', color='b', label='proj err')
            l1_err = self.results['l1']['proj_err']
            l2_err = self.results['l2']['proj_err']
            plt.hlines(l1_err, self.dependent_vars[0], self.dependent_vars[-1], "g", label='l1 Err')
            plt.hlines(l2_err, self.dependent_vars[0], self.dependent_vars[-1], "r", label='l2 Err')
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Projection Error")
            if save:
                fig.savefig(os.path.join(self.exp_dir, "proj_err_plt.png"))
            else:
                fig.show()
        if True:
            # plot l_hat for each 
            fig = plt.figure()
            self._set_plot()
            l_hats = [self.results[d]["l_bound"] for d in self.dependent_vars]
            plt.plot(self.dependent_vars, l_hats, '-o', label="Realized l")
            plt.xlabel(self.dependent_var_name)
            plt.ylabel("Bound on Realized Sketch Size")
            #plt.hlines(self.l1, self.dependent_vars[0], self.dependent_vars[-1], "g", label='l1')
            #plt.hlines(self.l2, self.dependent_vars[0], self.dependent_vars[-1], "r", label='l2')
            plt.legend(loc='best')
            if save:
                fig.savefig(os.path.join(self.exp_dir, "l_hat_plt.png"))
            else:
                fig.show()
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

class SketchExperiment(Experiment):
    SUPPORTED_SKETCHES = set(['jlt', 'cw', 'fd', 'pfd', 'batch-pfd'])
    #def __init__(self, exp_name, mat_fname, dependent_vars, dependent_var_name):
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
                _sketch_class_args = [self.sketch_types[sketch_t]['batch_size'], self.sketch_types[sketch_t]['alpha']]
            # now compute the sketches for each of the sketch sizes 
            self.results[sketch_t] = {}
            for l in self.sketch_sizes: 
                sketch_obj = _sketch_class(self.mat, l, *_sketch_class_args)
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

# maybe we should run tweaked on 2l rows? then it is comparable to our sketch on l rows? 
class TweakVsBatchPFDSketchExperiment(Experiment):
    """
    compare tweaked PFD with batched PFD for a range of alphas 
    goal is to show that our runtime is invariant to alpha, but theres is not 
    runs is number of times that we should repeat the experimeent for runtime 
    """
    def __init__(self, exp_name, mat_fname, l, alphas, runs=3, randomized=False):
        self.l = l
        self.alphas = alphas
        self.runs = runs
        self.randomized = randomized
        super(TweakVsBatchPFDSketchExperiment, self).__init__(exp_name, mat_fname, batch_sizes, "Alpha")

    def run_experiment(self):
        # compute sketches for each batch size, then save the results to a dictionary, 
        sketch_objs = []
        self.results = {"tweak": {}, "batch": {}}
        for a in self.alpha:
            times = []
            for i in self.runs:
                sketch_obj = BatchPFDSketch(self.mat, self.l, self.l, self.alpha, randomized=self.randomized)
                # compute the sketch 
                sketch_obj.compute_sketch()
                times.append(sketch_obj.sketching_time)
            self.results["batch"][a] = {"time": np.mean(times), 
                                        "err": sketch_obj.sketch_err(),
                                        "proj_err": sketch_obj.sketch_projection_err()}
            sketch_objs.append(sketch_obj)
            times = []
            for i in self.runs:
                #TODO: do we actually want 2 l here? 
                sketch_obj = TweakPFDSketch(self.mat, 2 * self.l, self.alpha, randomized=self.randomized)
                sketch_obj.compute_sketch()
                times.append(sketch_obj.sketching_time)
            sketch_obj.sketch = sketch_obj.sketch[:l, :]
            self.results["tweak"][a] = {"time": np.mean(times), 
                                        "err": sketch_obj.sketch_err(),
                                        "proj_err": sketch_obj.sketch_projection_err()}

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

def test_batch_exp():
    mat_fname = "med_svd_mat.txt"
    l = 30
    batch_sizes = np.arange(1, 2*l, max(l/5, 1))
    exp_name = "test_batch_exp"
    svd_batch_exp = BatchSketchExperiment(exp_name, mat_fname, l, batch_sizes, randomized=False)
    rand_batch_exp = BatchSketchExperiment(exp_name, mat_fname, l, batch_sizes, randomized=True)
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

if __name__ == "__main__":
    test_dyn_exp()
