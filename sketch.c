/*
 * Implementation of Algorithm 1 from: 
 * http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf
 */
#include <unistd.h>
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
    Matrix sketch, V, V_trunc, V_trunc_t, E;  
    sketch = zeros(l, mat->ncols);
    //TODO: dimensions are right here (as expected)
    //printf("Sketch dimensions: %d, %d\n", sketch.nrows, sketch.ncols);
    // svals will be the singular values of the sketch matrix
    // TODO: make this min(l, ncols) (and enforce l <= nrows)
    int svals_len = (int)fmin(mat->ncols, l);
    // TODO: need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: make V ncols X min(ncols, l)
    // TODO: should this be here or only in svd 
    V = zeros(mat->ncols, mat->ncols);

    // run while we have rows left to sketch 
    while(i < mat->nrows)
    {
        //printf("%d Sketch dimensions: %d, %d\n", i, sketch.nrows, sketch.ncols);
        //TODO: do we need to zero out svals, U, V each time? 
        zero_ind = zero_row(&sketch);
        if (zero_ind >=0)
        {
            //printf("Zero ind %d, i: %d\n", zero_ind, i);
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
            // TODO: does this matter            
            if (svd(&sketch, svals, &V, true) == 0)
            {
                V_trunc = truncate_cols_2(&V, svals_len);
                // should pick out the middle singular value
                del = svals[(int)floor(svals_len/2)];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                //printf("Singular values: ");
                for(j=0; j<svals_len; j++)
                {
                    //printf("%0.2f ", svals[j]);
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                    //if (svals[j] == 0.0)
                    //    printf("Zero Val at %d\n", j);
                }

                E = diag(svals, svals_len);                 
                //TODO: V is too large, we need to figure out which columns to keep?
                // reduced SVD: keep the first l columns, but how?
                //we could just change the ncols of V, while keeping the data there
                //printf("ncols: %d\n", V.ncols);
                V_trunc_t = transpose(&V_trunc); 
                // new sketch is EV^T
                // TODO: can optimize since E is diagonal
                sketch = mult(&E, &V_trunc_t);
            }
            else
            {
                printf("SVD Failed\n");
                exit(1);
            }
        }
    }
    // REPEAT ABOVE LOGIC to sketch
    // so why is our program so brutal? 
    // shoud we only do this if we don't have any zero rows at the end?
    zero_ind = zero_row(&sketch);
    if (zero_ind == -1)
    {
        if (svd(&sketch, svals, &V, true) == 0)
        {
            V_trunc = truncate_cols_2(&V, svals_len);
            // should pick out the middle singular value
            del = svals[(int)floor(l/2)];
            // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
            for(j=0; j<svals_len; j++)
            {
                svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                //if (svals[j] == 0.0)
                //    printf("Zero Val at %d\n", j);
            }

            E = diag(svals, svals_len);                 
            //TODO: V is too large, we need to figure out which columns to keep?
            // reduced SVD: keep the first l columns, but how?
            //we could just change the ncols of V, while keeping the data there
            V_trunc_t = transpose(&V_trunc); 
            // new sketch is EV^T
            // TODO: can optimize since E is diagonal
            sketch = mult(&E, &V_trunc_t);

        }
        else
        {
            printf("SVD Failed\n");
            exit(1);
        }
    }
    // TODO: clean up everything 
    free(svals);
    free_mat(&V);
    free_mat(&V_trunc);
    free_mat(&E);
    free_mat(&V_trunc_t);
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
int test_sketch(char* fname, int l, bool write, char* write_fname)
{
    Matrix mat = read_mat(fname);
    Matrix sketch = fd_sketch(&mat, l);
    if (write)
        write_mat(&sketch, write_fname); 

    double err = recon_error(&mat, &sketch);
    printf("Sq frobenius norm: %f\n", sq_frobenius_norm(&mat));
    double bound = (2.0 * sq_frobenius_norm(&mat)) / (float) l;
    printf("Err: %f, Bound: %f\n", err, bound);
    printf("Shape %d %d, L: %d\n", sketch.nrows, sketch.ncols, l);
    free_mat(&mat);
    free_mat(&sketch);
    if (bound >= err)
        return 1;
    else
        return 0;
}

// TODO: make this callable from the command line with the options
// read_file, write_file, l,  
//TODO: print V on the med_svd_mat 
// our version returns n X n V, but we want n X k
// keep first k columns 
int main(int argc, char* argv[])
{
    extern char *optarg;
    extern int optind;
    int c, err = 0;
    static char usage[] = "usage: -f fname [-w write_fname] -l l\n";
    char *fname, *write_fname;
    bool fflag, wflag, lflag = 0; 
    int l; 
    while((c = getopt(argc, argv, "f:w:l:")) != -1)
    {
        switch (c) 
        {
            case 'f':
                    fflag = 1;
                    fname = optarg;
                    break;
            case 'w':
                    wflag = 1;
                    write_fname = optarg;
                    break;
            case 'l':
                    lflag = 1;
                    l = atoi(optarg);
                    break;
        }
    }
    //printf("Command line args: %s %s %d\n", fname, write_fname, l);
    if (!(fflag && lflag))
    {
        printf("Parsing failed\n%s", usage);
        exit(1);
    }
    // read in a matrix, compute its sketch, then test accuracy?
    if (test_sketch(fname, l, wflag, write_fname))
        return 0;
    else
        return 1;    
}
