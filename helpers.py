"""
scripts to help run tests, generate and save matrices, test validity of operatiosn
"""
import numpy as np

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
        # save A to fname column major
        np.savetxt(fname, A.T) 

# what else do we need to script?
# test addition
# test multiplication
# test svd 
# test fd 
