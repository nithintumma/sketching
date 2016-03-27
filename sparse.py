import numpy as np 
import scipy.sparse as sps 

from fd_sketch import SparseBatchPFDSketch

# we want to build a sparse version of sketching function
# to do this, we need to iterate through the non-zero rows of the matrix (in sorted order preferably)
# once this works then we're almost done with the experiments
# what do we do about multithreading? we have to chunk up the data array of the matrices? 

# then lets start writing 


def main():
    a = sps.rand(100, 50, .2, format='coo')
    sketch = SparseBatchPFDSketch(a, 30, 30, 0.2, randomized=False)
    sketch.compute_sketch()
    print sketch.sketch_err()

# we want to store everything as csr, which we will use for sketching?? 
# that is the only thing that makes sense to me

if __name__=="__main__":
	main()
