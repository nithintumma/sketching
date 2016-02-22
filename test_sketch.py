import os 
import subprocess
import numpy as np
import time 

from helpers import load_matrix, write_matrix
import sys 
sys.path.append('../frequent-direction/')
from fd_sketch import sketch, calculateError, squaredFrobeniusNorm



# read in the matrix, compute the sketch and the error, compare to the error that we got 

# would be easiest to do all the scripting from python to run the C code 
# we should have the sketch program write the output sketch to file 
MATRIX_DIR = 'test_matrices'
RUN_SKETCH = './sketch'

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
    

if __name__ == '__main__':
    main()
