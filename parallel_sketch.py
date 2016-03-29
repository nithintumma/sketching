import time 
import os
from multiprocessing import Pool, current_process

import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmread

from fd_sketch import (SparseBatchPFDSketch, BatchPFDSketch, 
                        calculateError, calculate_projection_error)
from helpers import load_matrix 


def randomized_sketch(args):
	mat, l, b_size, alpha, sketch = args
	sketch_obj = BatchPFDSketch(mat, l, b_size, alpha, randomized=True)
	return sketch_obj.update_sketch(sketch)

def sketch(args):
	mat, l, b_size, alpha, sketch = args
	sketch_obj = BatchPFDSketch(mat, l, b_size, alpha, randomized=False)
	return sketch_obj.update_sketch(sketch)

def sparse_sketch(args):
	mat, l, b_size, alpha = args
	sketch_obj = SparseBatchPFDSketch(mat, l, b_size, alpha, randomized=False)
	return sketch_obj.compute_sketch()

def sparse_randomized_sketch(args):
	mat, l, b_size, alpha = args
	sketch_obj = SparseBatchPFDSketch(mat, l, b_size, alpha, randomized=True)
	return sketch_obj.compute_sketch()

# dont use pool
def parallel_bpfd_sketch(mat, l, alpha, batch_size, randomized=False, num_processes=2):
	if randomized:
		_sketch_func = randomized_sketch
	else:
		_sketch_func = sketch
	pool = Pool(processes=num_processes)
	# number of rows that should be assigned to each process (last one will get any extra)
	num_rows_per_p = mat.shape[0]/num_processes
	if num_rows_per_p == 0:
		raise Exception("Cannot have more processes than matrix rows")
	args = []
	for i in range(num_processes):
		# submatrix 
		start_ind = i*num_rows_per_p
		if i == (num_processes - 1):
			end_ind = mat.shape[0]		
		else:
			end_ind = (i+1) * num_rows_per_p
		args.append((mat[start_ind:end_ind, :],
					 l, batch_size, alpha, 
					 np.zeros((l, mat.shape[1]))
					 ))
	sketches = pool.map(_sketch_func, args)
	# now we want to merge the sketches 
	num_sketches = len(sketches)
	while num_sketches > 1:
		args = []
		stack_last = False 
		if (num_sketches % 2) == 1:
			stack_last = True
		for i in range(num_sketches/2):
			if (i == num_sketches/2 - 1) and stack_last:
				arg_tuple = (np.vstack((sketches[2*i], sketches[-1])), 
								l, batch_size, alpha, sketches[2*i+1])
			else:
				arg_tuple = (sketches[2*i], l, batch_size, alpha, sketches[2*i+1])
			args.append(arg_tuple)
		sketches = pool.map(_sketch_func, args)
		num_sketches = len(sketches)
	return sketches[0]

def sparse_parallel_bpfd_sketch(mat, l, alpha, batch_size, randomized=False, num_processes=2): 
	if randomized:
		_sparse_sketch_func = sparse_randomized_sketch
		_sketch_func = randomized_sketch
	else:
		_sparse_sketch_func = sparse_sketch
		_sketch_func = sketch
        print "Calling Sparse parallel"
	pool = Pool(processes=num_processes)
	unique_rows = np.unique(mat.row)
	row_inds, col_inds, data = mat.row, mat.col, mat.data 
	num_rows_per_p = len(unique_rows)/num_processes
	breakpoints = np.searchsorted(unique_rows, 
                            [i*num_rows_per_p for i in range(num_processes)])
	args = []
        # need csr to do row slicing 
        csr_mat = mat.tocsr()
        # lets assume that every row is non-zero for now
        print "About to construct the arguments"
        for i in range(num_processes):
            if i == num_processes - 1:
                s_ind = i*num_rows_per_p
                e_ind = csr_mat.shape[0] 
            else:
                s_ind = i*num_rows_per_p
                e_ind = (i+1)*num_rows_per_p
            submatrix = csr_mat[s_ind:e_ind, :] 
            print "Submatrix shape: ", submatrix.shape
            args.append((submatrix.tocoo(), l, batch_size, alpha))
	sketches = pool.map(_sparse_sketch_func, args)

	#for i in range(num_processes):
	#	if i == num_processes - 1:
        #            s_ind, e_ind = breakpoints[i], len(row_inds)
	#	else:
        #            s_ind, e_ind = breakpoints[i], breakpoints[i+1]
	#	# construct the sparse matrix
        #        submatrix = 
	#	submatrix = coo_matrix((data[s_ind:e_ind],
        #                                    (row_inds[s_ind:e_ind],
        #                                    col_inds[s_ind:e_ind])))
        #        print len(np.unique(row_inds[s_ind:e_ind]))
	#	print "Got here"
	#	print submatrix.shape
	#	raise Exception("failed on all counts")
	#	args.append((submatrix, l, batch_size, alpha))
        
        # this might be more efficient if we stack all the sketches and go in
        # one go
        print "Merging sketches"
	num_sketches = len(sketches)
	while num_sketches > 1:
		args = []
		stack_last = False 
		if (num_sketches % 2) == 1:
			stack_last = True
		for i in range(num_sketches/2):
			if (i == num_sketches/2 - 1) and stack_last:
				arg_tuple = (np.vstack((sketches[2*i], sketches[-1])), 
								l, 2*l, alpha, sketches[2*i+1])
			else:
				arg_tuple = (sketches[2*i], l, l, alpha, sketches[2*i+1])
			args.append(arg_tuple)
		sketches = pool.map(_sketch_func, args)
		num_sketches = len(sketches)
	return sketches[0]


def run_code():
    mat = load_matrix(mat_fname)
    print "Mat Shape: ", mat.shape
    l = 100
    alpha = 0.2
    batch_size = 100
    randomized=False
    num_processes=1
    print "Starting"
    start_time = time.time()
    sketch = parallel_bpfd_sketch(mat, l, alpha, batch_size, 
                                                                            randomized=randomized, num_processes=num_processes)
    sketching_time = time.time() - start_time 


    with open("experiments/parallel_results.txt", "a") as f:
            f.write("""Mat: %s, Rand: %r, l: %d, 
                        b: %d, alpha: %f, Processes: %d, Time: %f\n""" %(mat_fname, randomized, l, 
                                    batch_size, alpha, num_processes, sketching_time))
    print calculateError(mat, sketch)

# how do I test this? do I have an example file laying around somewhere? s
if __name__ == "__main__":
    mat_fname = 'ESOC.mtx'
    with open(os.path.join("test_matrices", mat_fname), "rb") as mfile:
        big_mat = mmread(mfile)
    mat = big_mat.tocoo()
    print "Mat Shape: ", mat.shape
    l = 200
    alpha = 0.2
    batch_size = 500
    randomized=False
    num_processes=2
    print "Starting with %d processes" % num_processes
    start_time = time.time()
    sketch = sparse_parallel_bpfd_sketch(mat, l, alpha, batch_size, 
                randomized=randomized, num_processes=num_processes)
    sketching_time = time.time() - start_time 
    with open("experiments/parallel_results.txt", "a") as f:
            f.write("""Mat: %s, Rand: %r, l: %d, 
                        b: %d, alpha: %f, Processes: %d, Time: %f\n""" %(mat_fname, randomized, l, 
                                    batch_size, alpha, num_processes, sketching_time))
