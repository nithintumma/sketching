/*
 * Implementation of Algorithm 1 from: 
 * http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"
#include "svd.h"
#include "sketch.h"

#define SQUARE(a)  a * a
#define FLOAT_EQ(a, b)      (fabs(a - b) < 0.0001)

// returns the index of the first zero row in mat, -1 if none exists
// TODO: return a list of all of the zero valued rows so that we can update the sketch faster 
int zero_row(Matrix* mat)
{
    bool flag; 
    for (int i=0; i<mat->nrows; i++)
    {
        flag = true;
        // return true if B has a zero valued row  
        for (int j=0; j<mat->ncols; j++)
        {
            // if every one is zero we want to break and return true
            if (!(FLOAT_EQ(get_ind(mat, i, j), 0.0)))
            {
                flag = false;
                break;
            }
        }
        if (flag)
            return i;
    }
    return -1;
}

// returns the sketch of the input matrix, processes it one row at a time 
Matrix fd_sketch(Matrix* mat, int l)
{
    // check that l is less than m
    if (l >= mat->nrows)
    {
        printf("Cannot sketch into more rows than input\n");
        exit(1);
    } 
    // would be nice if l was a multiple of 2 as well
    if (((l % 2) != 0) && l > 2)
    {
        printf("Require sketch size to be a multiple of 2, >= 4\n");
        exit(1);
    }
    int j, zero_ind;
    int i = 0;
    double del; 
    Matrix sketch, V, Vt, E;  
    sketch = zeros(l, mat->ncols);
    V = zeros(mat->ncols, mat->ncols);
    double* svals = malloc(sizeof(double) * mat->ncols);
    // run while we have rows left to sketch 
    while(i < mat->nrows)
    {
        //TODO: do we need to zero out svals, U, V each time? 
        zero_ind = zero_row(&sketch);
        if (zero_ind >=0)
        {
            // insert the current row of mat into a zero valued row of B
            //TODO: optimize this with a memcpy?
            for (j=0; j<mat->ncols; j++)
                set_ind(&sketch, zero_ind, j, get_ind(mat, i, j));
            i++;
        }
        else
        {
            // sketch has no zero valued rows, run sketching procedure
            // svd overwrites sketch with the U matrix
            if (svd(&sketch, svals, &V) == 0)
            {
                // should pick out the middle squared singular value 
                del = svals[(l-1)/2];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=0; j<mat->ncols; j++)
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                E = diag(svals, mat->ncols);                 
                Vt = transpose(&V); 
                // new sketch is EV^T
                sketch = mult(&E, &Vt);
            }
            else
            {
                printf("SVD Failed\n");
                exit(1);
            }
        }
    }
    return sketch;
}

double recon_error(Matrix* mat, Matrix* sketch)
{
    // return the reconstruction error using sketch 
    Matrix matt = transpose(mat);
    Matrix cov_mat = mult(&matt, mat);
    Matrix sketcht = transpose(sketch);
    Matrix cov_sketch = mult(&sketcht, sketch);
    Matrix diff = subtract(&cov_mat, &cov_sketch);
    return l2_norm(&diff);
}

// make sure that we satisfy the bound on reconstruction of covariance matrix
int test_sketch(char* fname, int l)
{
    Matrix mat = read_mat(fname);
    Matrix sketch = fd_sketch(&mat, l);
    double err = recon_error(&mat, &sketch);
    double bound = (2.0 * sq_frobenius_norm(&mat)) / (float) l;
    printf("Err: %f, Bound: %f\n", err, bound);
    if (bound >= err)
        return 1;
    else
        return 0;
}

int main(int argc, char* argv[])
{
    // read in a matrix, compute its sketch, then test accuracy?
    char* fname = "test_matrices/large_svd_mat.txt";
    int l = 40;
    if (test_sketch(fname, l))
        printf("Passed\n");
    else
        printf("Failed\n");
    return 0;
}
