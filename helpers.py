"""
scripts to help run tests, generate and save matrices, test validity of operatiosn
"""
import numpy as np
import os 
MATRIX_DIR = 'test_matrices/'

def gen_random_matrix(rows, cols, fname=None):
    """
    generate a random matrix with rows rows and cols columns
    if not fname, return matrix
    else, save to fname
    """
    A = np.random.rand(rows, cols)
    if not fname: 
        return A
    else: 
        # save A to fname row major
        write_matrix(A, fname)

def write_matrix(A, fname):
    """
    write numpy array to file in row order, space delimited
    write shape as first line (r c)
    """
    r, c = A.shape
    header = "%d %d" %(r, c)
    if MATRIX_DIR not in fname:
        fname = os.path.join(MATRIX_DIR, fname)
    np.savetxt(fname, A, delimiter=' ', fmt="%0.4f", header=header)

def load_matrix(fname):
    if MATRIX_DIR not in fname:
        fname = os.path.join(MATRIX_DIR, fname)
    return np.loadtxt(fname)
    
