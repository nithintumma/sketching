import random 
import numpy as np 
cimport numpy as np

cimport cython 

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DEF HASH_PRIME = 9223372036854775783

cdef long rand_param():
	return random.randint(0, HASH_PRIME-1)

# random numbers for hash functions 
cdef long a1 = rand_param()
cdef long a2 = rand_param()
cdef long a3 = rand_param()
cdef long a4 = rand_param()
cdef long a5 = rand_param()
cdef long a6 = rand_param()

cdef long col_hash(int i):
	return (a2 * i + a1) % HASH_PRIME

cdef int sign_hash(int i):
	cdef int val = (a6 * (i ** 3) + a5 * (i ** 2) + a4 * i  + a3) % 2
	return 2 * val - 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cw_sparse_sketch(np.float64_t [:, :] mat, np.float64_t [:, :] sketch, int l):
	cdef int nrows = mat.shape[0]
	cdef int ncols = mat.shape[1]
	# not sure if we can assert here 
	assert(sketch.shape[0] == l)
	assert(sketch.shape[1] == ncols)
	cdef int k, j, row_ind
	for k in range(nrows):
		row_ind = col_hash(k) % l
		sign = sign_hash(k)
		for j in range(ncols):
			sketch[row_ind, j] += sign * mat[k, j]
