import os 
import subprocess
import numpy as np
import time 
import re 

from helpers import load_matrix, write_matrix
import sys 
sys.path.append('../frequent-direction/')
from fd_sketch import sketch, calculateError, squaredFrobeniusNorm



# read in the matrix, compute the sketch and the error, compare to the error that we got 

# would be easiest to do all the scripting from python to run the C code 
# we should have the sketch program write the output sketch to file 
MATRIX_DIR = 'test_matrices'
RUN_SKETCH = './sketch'

def construct_sketches(orig_mat_fname, ls, check_c=True):
	"""
	generates sketch files using input l's 
	"""
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
	for l in ls:
		# generate p_sketch
		assert(l <= cols)
		p_sketch_pname = os.path.join(MATRIX_DIR, 
										"p_sketch_%d_%s" %(l, orig_mat_fname))
		start = time.time()
		p_sketch = sketch(mat, l)
		p_time = time.time() - start
		write_matrix(p_sketch, p_sketch_pname)
		p_fnames.append(p_sketch_pname)
		p_times.append(p_time)
		if check_c:
			c_sketch_pname = os.path.join(MATRIX_DIR, 
											"c_sketch_%d_%s" %(l, orig_mat_fname))
			subprocess.call(["make", "clean"])
			subprocess.call(["make", "sketch"])
			start = time.time()
			c_output = subprocess.check_output([RUN_SKETCH, '-f', mat_pname, '-w', c_sketch_pname, '-l', str(l)])
			c_time = time.time() - start
			print"C output on: ", l,  c_output
			c_fnames.append(c_sketch_pname)
			c_times.append(c_time)
	if check_c:
		return p_fnames, p_times, c_fnames, c_times
	else:
		return p_fnames, p_times 

def fd_bound(mat, l):
	return  2 * squaredFrobeniusNorm(mat) / l 	

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
			assert(np.isclose(p_err, c_err, atol=1e-2))
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

def main():
	mat_name = "large_svd_mat.txt"
	# get m, n s
	l = 300
	err_bound, f_norm, p_err, p_time, c_err, c_time = test(mat_name, l, check_c=True)
	print "Sq Frobenius Norm %f" % (f_norm)
	print "Py Error %f, Bound: %f, Passed: %r, Took: %f s" % (p_err, err_bound, p_err < err_bound, p_time)
	if c_err:
		print "C Error %f, , Bound: %f, Passed %r, Took: %f s" % (c_err, err_bound, c_err < err_bound, c_time)
 

def experiment_1():
	ls = [10, 20, 30, 40]
	fname = "med_svd_mat.txt"
	check_c = True
	if check_c:
		p_fnames, p_times, c_fnames, c_times = construct_sketches(fname, ls, check_c=True)
	else:
		p_fnames, p_times = construct_sketches(fname, ls, check_c=True)
	# if we get here good job
	print "Constructed sketches"
	print plot_errors(fname, p_fnames, c_fnames)

	# calculate errors 

if __name__ == '__main__':
    experiment_1()
