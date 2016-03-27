"""
scripts to help run tests, generate and save matrices, test validity of operatiosn
"""
import numpy as np
import os 
import cPickle as pickle
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

# load CIFAR matrix (pickle) and store it in test_matrices
def get_cifar_matrix(fname, path='../../data/cifar-10-batches-py'):
    if not os.path.exists(path):
        raise Exception("Path does not exist %s" %path)
    fo = open(os.path.join(path, fname), "rb")
    dict = pickle.load(fo)
    fo.close()
    write_matrix(dict["data"], fname)

def get_all_cifar_matrices(path='../../data/cifar-10-batches-py'):
    fname_pre = "data_batch_"
    mat = None
    for i in range(1, 6):
        fname = "%s%d"  %(fname_pre, i)
        f_path = os.path.join(path, fname)
        fo = open(f_path, "rb")
        d = pickle.load(fo)
        if mat is None:
            mat = d['data']
        else:
            mat = np.vstack((mat, d['data']))
    print mat.shape
    write_matrix(mat, "cifar_data")

def aggregate_cifar_matrice(cifar_path='../../data/'):
    fname_pre = "data_batch_"
    path = os.path.join(cifar_path, 'cifar-10-batches-py')
    mat = None
    for i in range(1, 6):
        fname = "%s%d"  %(fname_pre, i)
        f_path = os.path.join(path, fname)
        fo = open(f_path, "rb")
        d = pickle.load(fo)
        if mat is None:
            mat = d['data']
        else:
            mat = np.vstack((mat, d['data']))
    path = os.path.join(cifar_path, "cifar-100-python")    
    fo = open(os.path.join(path, "train"), "rb")
    d = pickle.load(fo)
    mat = np.vstack((mat, d['data']))
    return mat

 
    
