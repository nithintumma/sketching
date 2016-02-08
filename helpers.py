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
        write_matrix(A, os.path.join(MATRIX_DIR, fname))

def write_matrix(A, fname):
    """
    write numpy array to file in row order, space delimited
    write shape as first line (r c)
    """
    r, c = A.shape
    header = "%d %d" %(r, c)
    np.savetxt(os.path.join(MATRIX_DIR, fname), 
                    A, delimiter=' ', fmt="%0.4f", header=header)

def load_matrix(fname):
    return np.loadtxt(os.path.join(MATRIX_DIR, fname))

#TODO: construct large random test matrices and write them
#TODO: also write their sum, product, transpose, scal_mult, etc. for testing
