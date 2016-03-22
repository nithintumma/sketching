#!/usr/bin/env python
import numpy as np
import numpy.linalg as ln
import math
import sys
import time 
import random

import os
import subprocess 

from helpers import load_matrix, write_matrix 

from fbpca import pca as rand_svd 

import sys
libskylark_path = os.path.join("..", "libskylark/")
sys.path.append(libskylark_path)

# cython code for CW-13 Sparse Sketch 
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from sketch import cw_sparse_sketch


"""
additions to Python FD sketching procedure: batched version of below 
"""
class Sketch(object):
    """
    helper functions that can be used by all sketches 
    """
    def write_sketch(write_fname=None):
        if self.sketch is None:
            raise Exception("Cannot write sketch until it is computed")
        if write_fname is None:
            # default name is sketch_nrows_ncols_time.txt
            write_fname = "sketch_%d_%d_%d.txt" %(self.sketch.shape[0], self.sketch.shape[1], int(time.time()))
            write_matrix(self.sketch, write_fname)

    def load_sketch(sketch_fname):
        self.sketch = load_matrix(sketch_matrix)
        self.sketching_time = 0

    def sketch_err(self):
        if self.sketch is None:
            self.compute_sketch()
        return calculateError(self.mat, self.sketch)

    def sketch_projection_err(self):
        if self.sketch is None:
            self.compute_sketch()
        return calculate_projection_error(self.mat, self.sketch)

class BatchFDSketch(Sketch):
    def __init__(self, mat, l, b_size, randomized=False, track_del=False):
        self.mat = mat
        self.m = mat.shape[1]
        self.l = l
        self.b_size = b_size
        self.track_del = track_del
        self.delta = 0.0 
        if math.floor(self.l) >= self.m:
            raise ValueError('Error: l must be smaller than m')
        if self.l >= self.mat.shape[0]:
            raise ValueError('Error: l must not be greater than n')
        self.randomized = randomized
        if self.randomized:
            self._sketch_func = self._rand_svd_sketch
        else:
            self._sketch_func = self._svd_sketch
        self.sketch = None

    def _svd_sketch(self, mat_b):
        mat_u, vec_sigma, mat_vt = ln.svd(mat_b, full_matrices=False) 
        if (self.l + self.b_size > self.m):
            # then vec_sigma, mat_vt will be m, m X m respectively, we need to make them larger 
            extra_rows = self.l + self.b_size - self.m 
            vec_sigma = np.hstack((vec_sigma, np.zeros(extra_rows)))
            mat_vt = np.vstack((mat_vt, np.zeros((extra_rows, self.m))))
        # obtain squared singular value for threshold
        squared_sv_center = vec_sigma[self.l-1] ** 2
        if self.track_del:
            self.delta = self.delta + squared_sv_center
        # update sigma to shrink the row norms
        sigma_tilda = [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)]
        # update matrix B where at least half rows are all zero
        return np.dot(np.diagflat(sigma_tilda), mat_vt)

    def _rand_svd_sketch(self, mat_b):
        # use fbpca rand_svd (PCA) function to approximate PCA 
        # only want first l values, 
        # do we care about block size for power iteration method? 
        mat_u, vec_sigma, mat_vt = rand_svd(mat_b, self.l, raw=True)
        # need to return an (l + b) X ncols matrix, so add b rows of zero to result 
        extra_rows = self.b_size 
        vec_sigma = np.hstack((vec_sigma, np.zeros(extra_rows)))
        mat_vt = np.vstack((mat_vt, np.zeros((extra_rows, self.m))))
        squared_sv_center = vec_sigma[self.l-1] ** 2
        if self.track_del:
            self.delta = self.delta + squared_sv_center
        sigma_tilda = [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)]
        return np.dot(np.diagflat(sigma_tilda), mat_vt)

    def compute_sketch(self):
        start_time = time.time()
        if self.sketch is not None:
            return self.sketch
        mat_b = np.zeros([self.l + self.b_size, self.m])
        # compute zero valued row list
        zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
        # repeat inserting each row of matrix A 
        for i in range(0, self.mat.shape[0]):
            # insert a row into matrix B
            mat_b[zero_rows[0], :] = self.mat[i, :]
            # remove zero valued row from the list
            zero_rows.remove(zero_rows[0])
            # if there is no more zero valued row
            if len(zero_rows) == 0:
                # compute SVD of matrix B, we want to find the first l
                mat_b  = self._sketch_func(mat_b)
                # update the zero valued row list
                zero_rows = np.nonzero([round(s, 7) == 0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
        # why do we need this here? 
        # do we need to do a sketch one last time at the end? 
        mat_b = self._sketch_func(mat_b)
        # get rid of extra non-zero rows when we return 
        self.sketch = mat_b[:self.l, :]
        self.sketching_time = time.time() - start_time
        return self.sketch

class FDSketch(BatchFDSketch):
    """"
    Fast FD Sketch, can be implemented as non-randomized batched version with l = l'/2, batch_size = l'/2
    In our versions, l actually corresponds to the number of non-zero rows of the sketch, so we don't divide by 2
    """
    def __init__(self, mat, l):
        super(FDSketch, self).__init__(mat, l, l, randomized=False)

class DynamicFDSketch(BatchFDSketch):    
    """
    begin sketching with l1, after t (changepoint) rows have been added 
    sketch with l2 rows 
    """
    def __init__(self, mat, l1, l2, t, b_size, randomized=False):
        self.l1 = l1
        self.l2 = l2
        # point at which we will change sketch sizes 
        self.t = t
        assert(self.t < mat.shape[0])
        assert(self.l1 + b_size > self.l2)
        # holds the mass subtracted before/after changepoint
        self.del_1 = 0.0
        self.del_2 = 0.0
        super(DynamicFDSketch, self).__init__(mat, l1, b_size, randomized=randomized, track_del=True)

    def compute_sketch(self):
        # assumes that we are tracking delta (self.track_del == True)
        if self.sketch is not None:
            return self.sketch
        start_time = time.time()
        if self.sketch is not None:
            return self.sketch
        mat_b = np.zeros([self.l + self.b_size, self.m])
        # compute zero valued row list
        zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
        # repeat inserting each row of matrix A 
        for i in range(0, self.mat.shape[0]):
            # insert a row into matrix B
            if i == self.t:
                self.l = self.l2
                # do we also need to add zero rows to the sketch? probably 
                num_new_rows = self.l2 + self.b_size - mat_b.shape[0]
                mat_b = np.vstack((mat_b, np.zeros((num_new_rows, mat_b.shape[1]))))
                zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
                self.del_1 = self.delta
                self.delta = 0.0
            mat_b[zero_rows[0], :] = self.mat[i, :]
            # remove zero valued row from the list
            zero_rows.remove(zero_rows[0])
            # if there is no more zero valued row
            if len(zero_rows) == 0:
                # compute SVD of matrix B, we want to find the first l
                mat_b  = self._sketch_func(mat_b)
                # update the zero valued row list
                zero_rows = np.nonzero([round(s, 7) == 0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
        # why do we need this here? 
        # do we need to do a sketch one last time at the end? 
        mat_b = self._sketch_func(mat_b)
        # get rid of extra non-zero rows when we return 
        self.sketch = mat_b[:self.l, :]
        self.del_2 = self.delta
        self.sketching_time = time.time() - start_time
        return self.sketch

class BatchPFDSketch(BatchFDSketch):
    """
    Batched PFD Sketch,  
    :param mat original matrix (numpy array)
    :param l: sketch size 
    :param batch_size: batch size (number of rows to be added to sketch before sketching procedure)
    :param alpha (\in (0, 1]) (fraction of singular values that will be subtracted)
    """
    def __init__(self, mat, l, batch_size, alpha, randomized=False):
        assert(alpha > 0 and alpha <= 1)
        self.alpha = alpha 
        # FOR FAST-PFD, del_ind = l(2 - \alpha) - 1, ind of singular value to subtract 
        #self.del_ind = max(0, math.floor(l * (2.0 - self.alpha) - 1))
        self.del_ind = l - 1 
        # FOR FAST-PFD alpha_ind = 2l(1 - \alpha), ind of singular value to begin subtracting from 
        #self.alpha_ind = max(0, math.floor(2 * l * (1.0 - self.alpha)))
        self.alpha_ind = min(math.floor(l * (1.0 - self.alpha)), l - 1)
        super(BatchPFDSketch, self).__init__(mat, l, batch_size, randomized=randomized)

    # override _svd_sketch and _random_svd_sketch to use del_ind and alpha_ind
    def _svd_sketch(self, mat_b):
        mat_u, vec_sigma, mat_vt = ln.svd(mat_b, full_matrices=False) 
        if (self.l + self.b_size > self.m):
            # then vec_sigma, mat_vt will be m, m X m respectively, we need to make them larger 
            extra_rows = self.l + self.b_size - self.m 
            vec_sigma = np.hstack((vec_sigma, np.zeros(extra_rows)))
            mat_vt = np.vstack((mat_vt, np.zeros((extra_rows, self.m))))
        # obtain squared singular value for threshold
        squared_sv_center = vec_sigma[self.del_ind] ** 2
        # update sigma to shrink the row norms, only subtract from alpha_ind to end of vector 
        sigma_tilde = list(vec_sigma[:self.alpha_ind]) + [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)[self.alpha_ind:]]
        # update matrix B where at least half rows are all zero
        return np.dot(np.diagflat(sigma_tilde), mat_vt)

    # override _random_svd_sketch to use del_ind and alpha_ind
    def _rand_svd_sketch(self, mat_b):
        # use fbpca rand_svd (PCA) function to approximate PCA 
        # do we care about block size for power iteration method? 
        mat_u, vec_sigma, mat_vt = rand_svd(mat_b, self.l, raw=True)
        # need to return an (l + b) X ncols matrix, so add b rows of zero to result 
        extra_rows = self.b_size 
        vec_sigma = np.hstack((vec_sigma, np.zeros(extra_rows)))
        mat_vt = np.vstack((mat_vt, np.zeros((extra_rows, self.m))))
        squared_sv_center = vec_sigma[self.del_ind] ** 2
        sigma_tilde = list(vec_sigma[:self.alpha_ind]) + [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)[self.alpha_ind:]]
        return np.dot(np.diagflat(sigma_tilda), mat_vt)

class PFDSketch(BatchPFDSketch):
    """
    Fast PFD Sketch
    :param mat original matrix (numpy array)
    :param l: sketch size 
    :param batch_size: batch size (number of rows to be added to sketch before sketching procedure)
    """
    def __init__(self, mat, l, alpha):
        super(PFDSketch, self).__init__(mat, l, l, alpha, randomized=False)

class CWSparseSketch(Sketch):
    """
    From Theorem 2.6 of http://researcher.watson.ibm.com/researcher/files/us-dpwoodru/journal.pdf
    Constructs a sparse sketching matrix and applies to input matrix 
    """
    # taken from example count-min sketch at: https://tech.shareaholic.com/2012/12/03/the-count-min-sketch-how-to-count-over-large-keyspaces-when-about-right-is-good-enough/
    HASH_PRIME = 9223372036854775783
    def __init__(self, matrix, l, use_hash=False):
        """
        matrix is input matrix
        l is sketch size
        """
        self.mat = matrix
        self.l = l
        self.use_hash = use_hash
        # use current time as seed
        np.random.seed(int(time.time()))
        if self.use_hash:
            self._col_hash = self._generate_pairwise_hash()
            self._sign_hash = self._generate_fourwise_hash()

    # used for hash funcs 
    def _random_paramater(self):
        return np.random.randint(0, HASH_PRIME - 1)

    # could we do a faster implementations
    def _generate_pairwise_hash(self):
        # used to determine which row is non-zero per column 
        a, b = self._random_paramater(), self._random_paramater()
        return lambda x: (a * x + b) % HASH_PRIME % self.l

    def _generate_fourwise_hash(self):
        # used as sign hash
        a, b, c, d = self._random_paramater(), self._random_paramater(), self._random_paramater(), self._random_paramater()
        def _hash(x):
            sign = (a * (x**3) + b * (x ** 2) + c * (x) + d) % HASH_PRIME % 2
            # alternatively we can return 2 * val - 1
            if sign == 0:
                sign = -1
            return sign
        return _hash

    def _random_sign(self):
        # choose a random sign func
        return random.choice([-1, 1])

    def _random_row(self):
        # choose a random row from the sketch (which is l X self.mat.nrows)
        return np.random.randint(0, self.l)

    # DON'T NEED THIS 
    def _compute_sketching_matrix(self):
        # store the sketching matrix as a list of tuples (one per col), where the tuple is (row_index, sign), sign is 1 OR -1
        if self.use_hash:
            self.sketching_matrix = [(self._col_hash(i), self._sign_hash(i)) for i in range(self.mat.shape[1])]
        # what if we just pick the columns randomly uniformly 
        else: 
            self.sketching_matrix = [(self._random_row(), self._random_sign()) for i in range(self.mat.shape[1])]

    def compute_sketch(self, use_cython=True):
        # can efficiently compute multiplication because only one column is non-zero
        if use_cython:
            self.compute_sketch_cython()
        else:
            start_time = time.time()
            sketch = np.zeros((self.l, self.mat.shape[1]))
            for k in range(self.mat.shape[0]):
                for j in range(self.mat.shape[1]):
                    if self.use_hash: 
                        row_ind, sign = self._col_hash(k), self._sign_hash(k)
                    else:
                        row_ind, sign = self._random_row(), self._random_sign()
                    sketch[row_ind, j] += sign * self.mat[k, j]
            self.sketch = sketch
            self.sketching_time = time.time() - start_time

    def compute_sketch_cython(self):
        start_time = time.time()
        sketch = np.zeros((self.l, self.mat.shape[1]))
        # call cython function, which should modify sketch in place 
        cw_sparse_sketch(self.mat, sketch, self.l)
        self.sketch = sketch
        self.sketching_time = time.time() - start_time


class JLTSketch(Sketch):
    """
    uses random gaussian matrix to compute sketch 
    """
    def __init__(self, matrix, l):
        self.mat = matrix
        self.l = l

    def compute_sketch(self):
        # use sklearn built in random projection matrix?? but does this let us constrol how big the matrix will be? 
        start_time = time.time()
        self.sketching_matrix = np.random.normal(size=(self.l, self.mat.shape[0]))
        self.sketch = np.dot(self.sketching_matrix, self.mat)
        self.sketching_time = time.time() - start_time

# functions to return sketches from classes above 
def batch_fd_sketch(mat, l, b_size, randomized=False):
    b_sketch = BatchFDSketch(mat, l, b_size, randomized)
    return b_sketch.compute_sketch()

def dynamic_fd_sketch(mat, l1, l2, t, b_size, randomized=False):
    db_sketch = DynamicFDSketch(mat, l1, l2, t, b_size, randomized)
    return db_sketch.compute_sketch()

def pdf_sketch(mat, l, alpha):
    return None


""" This is a simple and deterministic method for matrix sketch.
The original method has been introduced in [Liberty2013]_ .

[Liberty2013] Edo Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013.
"""

def test_sparse_sketch():
    test_mat = np.eye(20)
    l = 10
    sketch_obj = CWSparseSketch(test_mat, l)
    sketch_obj.compute_sketch()
    print sketch_obj.sketch

# from github project: https://github.com/hido/frequent-direction/
def fd_sketch(mat, l):
    """Compute a sketch matrix of input matrix 
    Note that l must be smaller than m
    
    :param mat: original matrix to be sketched (n x m)
    :param ell: the number of rows in sketch matrix
    :returns: sketch matrix (l x m)
    """

    # number of columns
    m = mat.shape[1]

    # Input error handling
    if l >= m:
        raise ValueError('Error: ell must be smaller than m')
    if l >= mat.shape[0]:
        raise ValueError('Error: ell must not be greater than n')

    def svd_sketch(mat_b):
        mat_u, vec_sigma, mat_vt = ln.svd(mat_b, full_matrices=False)
        # obtain squared singular value for threshold
        squared_sv_center = vec_sigma[l-1] ** 2
        # update sigma to shrink the row norms
        sigma_tilda = [(0.0 if d < 0.0 else math.sqrt(d)) for d in (vec_sigma ** 2 - squared_sv_center)]
        # update matrix B where at least half rows are all zero
        return np.dot(np.diagflat(sigma_tilda), mat_vt)

    # initialize output matrix B
    mat_b = np.zeros([2 * l, m])
    # compute zero valued row list
    zero_rows = np.nonzero([round(s, 7) == 0.0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
    # repeat inserting each row of matrix A
    for i in range(0, mat.shape[0]):
        # insert a row into matrix B
        #print "Zero row ", zero_rows[0]
        mat_b[zero_rows[0], :] = mat[i, :]
        # remove zero valued row from the list
        zero_rows.remove(zero_rows[0])
        # if there is no more zero valued row
        if len(zero_rows) == 0:
            mat_b = svd_sketch(mat_b)
            # update the zero valued row list
            zero_rows = np.nonzero([round(s, 7) == 0 for s in np.sum(mat_b, axis = 1)])[0].tolist()
    # sketch last rows, and return l-sized sketch
    mat_b = svd_sketch(mat_b)
    return mat_b[:l, :]

#TODO: change name to sq_frobenius_norm 
def squaredFrobeniusNorm(mat):
    """Compute the squared Frobenius norm of a matrix

    :param mat: original matrix
    :returns: squared Frobenius norm
    """
    return ln.norm(mat, ord = 'fro') ** 2

#TODO: change this name to calculate_covariance_err 
def calculateError(mat, mat_b, normalized=True):
    """
    Compute the degree of error by sketching
    :param mat: original matrix
    :param mat_b: sketch matrix
    :returns: covariance reconstruction error
    """
    dot_mat = np.dot(mat.T, mat)
    dot_mat_b = np.dot(mat_b.T, mat_b)
    if normalized:
        return (ln.norm(dot_mat - dot_mat_b, ord = 2) / squaredFrobeniusNorm(mat))
    else:
        return ln.norm(dot_mat - dot_mat_b, ord = 2)

# calculates orthogonal projection matrix onto A 
def projection_matrix(A):
    # projection At (A At)^-1 A
    try:
        inv_mat = np.linalg.inv(np.dot(A, A.T))
    except:
        inv_mat = np.linalg.pinv(np.dot(A, A.T))
    return np.dot(A.T, np.dot(inv_mat, A))

def calculate_projection_error(mat, sketch, k=None, normalized=True):
    """
    mat: original matrix
    sketch: sketch matrix
    """
    if k is None:
        k = mat.shape[1]/2
    # take the SVD of sketch, 0 out the k+1 - last singular values, multiply U, S', Vt
    sU, ssvals, sVt = np.linalg.svd(sketch, full_matrices=False)
    ssvals[k:] = 0.0
    # top k scaled singular vectors of sketch
    sketch_k = np.dot(sU, np.dot(np.diag(ssvals), sVt))
    # project mat onto row space of sketch_K
    proj_sketch_k = projection_matrix(sketch_k)
    proj_A = np.dot(mat, proj_sketch_k) 
    if not normalized:
        return squaredFrobeniusNorm(mat - proj_A)
    else:
        # |A - A_k |_F^2 = sum of k + 1 -> last squared singular values 
        svals = np.linalg.svd(mat, compute_uv=False)
        return (squaredFrobeniusNorm(mat - proj_A) / np.sum(svals[k:] ** 2))
