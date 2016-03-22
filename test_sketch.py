import os 
import subprocess
import numpy as np
import time 
import re 
import matplotlib.pyplot as plt 

from helpers import load_matrix, write_matrix
from fd_sketch import BatchFDSketch, DynamicFDSketch, calculateError, squaredFrobeniusNorm

# CONSTANTS 
MATRIX_DIR = 'test_matrices'
RUN_SKETCH = './sketch'

def fd_rank_bound(mat, l, k):
	1.0/(l - k)
	U, s_vals, Vt = np.linalg.svd(mat)
	ss_vals = s_vals ** 2
	return (1.0/(l - k)) * (np.sum(ss_vals[k:]))

def fd_bound(mat, l):
	return  2 * squaredFrobeniusNorm(mat) / l 	

def run_fd_sketch(mat_pname, write_pname, l):
	# do exactly what the C code is doing (including reading and writing files)
	mat = load_matrix(mat_pname)
	p_sketch = sketch(mat, l)
	write_matrix(p_sketch, write_pname)
	return p_sketch

def construct_sketches(orig_mat_fname, ls, check_c=True, force_comp=False):
	"""
	generates sketch files using input l's 
	@param orig_mat_fname: name of input matrix file inside MATRIX_DIR
	@param ls: List of sketch sizes to construct
	@param check_c: flag whether to construct sketches using custom C implementation
	@param force_comp: disregard cached sketch, recompute (required to determine runtime)
	"""
	# TOOD: check if we've already constructed it. 
	# we can check for the right filename and then some
	# maybe don'r run again? 
	p_fnames = []
	p_times = []
	if check_c:
		c_fnames = []
		c_times = []
	if ".txt" not in orig_mat_fname:
		orig_mat_fname += ".txt"
	mat_pname = os.path.join(MATRIX_DIR, orig_mat_fname)
	mat = load_matrix(mat_pname)
	rows, cols = mat.shape
	# create C binary 
	if check_c:
		subprocess.call(["make", "clean"])
		subprocess.call(["make", "sketch"])

	for l in ls:
		# generate p_sketch
		assert(l <= cols)
		p_sketch_pname = os.path.join(MATRIX_DIR, 
										"p_sketch_%d_%s" %(l, orig_mat_fname))
		if not(check_sketch_file(mat_pname, p_sketch_pname, l)) or force_comp:
			start = time.time()
			run_fd_sketch(mat_pname, p_sketch_pname, l)
			p_time = time.time() - start
		else:
			p_time = 0.0
		p_fnames.append(p_sketch_pname)
		p_times.append(p_time)
		if check_c:
			c_sketch_pname = os.path.join(MATRIX_DIR, 
											"c_sketch_%d_%s" %(l, orig_mat_fname))
			
			#
			if not(check_sketch_file(mat_pname, c_sketch_pname, l)) or force_comp:
				start = time.time()
				# not a fair comparison because of shit
				c_output = subprocess.check_output([RUN_SKETCH, '-f', mat_pname, '-w', c_sketch_pname, '-l', str(l)])
				c_time = time.time() - start				
				# so we should get the time from c_output, hopefully just one float shows up
				#num_matches = re.findall("\d+\.\d+", c_output)
				#c_time = float(num_matches[0])
				print"C output on: ", l,  c_output
				print "Python time: ", c_time
			else:
				c_time = 0.0
			c_fnames.append(c_sketch_pname)
			c_times.append(c_time)
	if check_c:
		return p_fnames, p_times, c_fnames, c_times
	else:
		return p_fnames, p_times 

def check_sketch_file(orig_mat_fname, sketch_fname, l):
	# assert os.path.exists
	assert(os.path.exists(orig_mat_fname))
	mat = load_matrix(orig_mat_fname)
	if not os.path.exists(sketch_fname):
		return False
	sketch = load_matrix(sketch_fname)
	if mat.shape[1] != sketch.shape[1]:
		return False
	bound = fd_bound(mat, l)
	err = calculateError(mat, sketch)
	if err < bound: 
		return True 
	else:
		return False

def plot_errors(orig_mat_fname, p_fnames, c_fnames = None):
	# load the original matrix 
	if ".txt" not in orig_mat_fname:
		orig_mat_fname += ".txt"
	mat_pname = os.path.join(MATRIX_DIR, orig_mat_fname)
	mat = load_matrix(mat_pname)
	rows, cols = mat.shape
	ls = []
	errs = []
	bounds = []
	regex = re.compile('\d+')
	if c_fnames:
		for p_fname, c_fname in zip(p_fnames, c_fnames):
			# extract the sketch size
			# calculate the error from each 
			# assert t
			p_l = int(regex.search(p_fname).group(0))
			c_l = int(regex.search(c_fname).group(0))
			assert(p_l == c_l)
			p_sketch = load_matrix(p_fname)
			c_sketch = load_matrix(c_fname)
			assert(p_sketch.shape == c_sketch.shape)
			bound = fd_bound(mat, p_l)
			p_err = calculateError(mat, p_sketch)
			c_err = calculateError(mat, c_sketch)
			print p_err, c_err
			# the optimizations make floating point arithmetic not exact 
			assert(np.isclose(p_err/c_err, 1.0, atol=1e-1))
			ls.append(p_l)
			errs.append(p_err)
			bounds.append(bound)
	else:
		for p_fname in p_fnames:
			p_l = int(regex.search(p_fname).group(0))
			p_sketch = load_matrix(p_fname)
			bound = fd_bound(mat, p_l)
			p_err = calculateError(mat, p_sketch)
			ls.append(p_l)
			errs.append(p_err)
			bounds.append(bound)
	# lets plot this shit 
	plt.plot(ls, errs, '-o', color='b', label='cov err')
	plt.plot(ls, bounds, '-o', color='r', label='upper bound')
	plt.xlabel("Sketch size (l)")
	plt.ylabel("Error")
	title = "Sketch size vs Reconstruction Error: %d X %d" %(mat.shape[0], mat.shape[1])
	plt.title(title)
	plt.grid()
	plt.legend(loc=3)
	plt.yscale('log')
	#plt.xlim(25, 375)
	plt.show() 
	return ls, errs, bounds

def test(mat_name, l, check_c = True):
	"""
	@param mat_name: name of matrix file
	@param l: number of columns in sketch
	"""
	if ".txt" not in mat_name:
		mat_name += ".txt"
	mat_pname = os.path.join(MATRIX_DIR, mat_name)
	mat = load_matrix(mat_pname)
	print "Original shape: %r, l: %d" %(mat.shape, l)
	f_norm = squaredFrobeniusNorm(mat)
	start = time.time()
	p_sketch = sketch(mat, l)
	p_time = time.time() - start
	print "Sketch shape: ", p_sketch.shape
	p_err = calculateError(mat, p_sketch)
	# calculate bound on error
	err_bound = 2 * squaredFrobeniusNorm(mat) / l 
	if not check_c:
		return err_bound, f_norm, p_err, None

	# run sketch.c on the matrix 
	sketch_pname = os.path.join(MATRIX_DIR, "sketch_" + mat_name)
	subprocess.call(["make", "clean"])
	subprocess.call(["make", "sketch"])
	# need to check output 
	c_start = time.time()
	err = subprocess.check_output([RUN_SKETCH, '-f', mat_pname, '-w', sketch_pname, '-l', str(l)])
	c_time = time.time() - c_start
	print "Sketch output: ", err
	c_sketch = load_matrix(sketch_pname)
	assert (c_sketch.shape == p_sketch.shape)
	c_err = calculateError(mat, c_sketch)
	return err_bound, f_norm, p_err, p_time, c_err, c_time

def test_svd(mat_name):
	if ".txt" not in mat_name:
		mat_name += ".txt"
	mat_pname = os.path.join(MATRIX_DIR, mat_name)
	mat = load_matrix(mat_pname)
	U, w, Vt = np.linalg.svd(mat, full_matrices=False)
	V = Vt.T
	print "V shape: ", V.shape
	print np.around(V, 2)
	print "Singular values: ", w

def experiment_1():
	# 100, 200, ... 800, 900
	ls = np.arange(100, 1000, 100)
	fname = "2000_1000_mat.txt"
	check_c = True
	force_comp = True
	if check_c:
		p_fnames, p_times, c_fnames, c_times = construct_sketches(fname, ls, check_c=check_c, force_comp=force_comp)
	else:
		p_fnames, p_times = construct_sketches(fname, ls, check_c=check_c)
	# if we get here good job
	print "Constructed sketches"
	print plot_errors(fname, p_fnames, c_fnames)
	# plot the runtimes  
	plt.figure()
	print "P times: ", p_times
	print "C times: ", c_times
	plt.plot(ls, np.array(c_times)/np.array(p_times), '-o', color='b', label='C/Numpy Time')
	#plt.plot(ls, c_times, '-o', color='r', label='C Time')
	plt.xlabel('Sketch Size (l)')
	plt.ylabel('Runtime (s)')
	title = "Sketch Size vs Runtime, Numpy & C"
	plt.title(title)
	plt.grid()
	plt.legend(loc=3)
	plt.show()


def batch_experiments():
	# how can we abstract out the experiment class to make this stuff efficient/useful
	return None


# we have that for two things

if __name__ == '__main__':
    experiment_1()
