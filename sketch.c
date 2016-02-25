/*
 * Implementation of Algorithm 1 from: 
 * http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf
 */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#include <Accelerate/Accelerate.h>
#include "matrix.h"
#include "svd.h"
#include "svd_lapack.h"
#include "sketch.h"

#define SQUARE(a)  a * a
#define FLOAT_EQ(a, b)      (fabs(a - b) < 0.0001)

// returns the index of the first zero row in mat, -1 if none exists
// TODO: return a list of all of the zero valued rows so that we can update the sketch faster 
int zero_row(Matrix* mat)
{
    bool flag; 
    // don't ever break out of this and instead once we reach a zero row, 
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

//TODO: change the order of for loops around for col order?
int zero_row_2(Matrix* mat, int* zero_rows)
{
    // zero_rows is an array of size mat->nrows, will hold indices for zero row
    int num_zero_rows = 0;
    bool flag; 
    // don't ever break out of this and instead once we reach a zero row, 
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
        {
            zero_rows[num_zero_rows] = i;
            num_zero_rows++;
        }
    }
    return num_zero_rows;
}

// use lapack svd to compute the FD sketch
Matrix fd_sketch_l(Matrix* mat, int l)
{
    // check that l is less than m
    if (l >= mat->nrows)
    {
        printf("Cannot sketch into more rows than input\n");
        exit(1);
    } 
    // would be nice if l was a multiple of 2 as well
    if (((l % 2) != 0) && l > 2 && (l <= mat->nrows))
    {
        printf("Require sketch size to be a multiple of 2, >= 4, > nrows\n");
        exit(1);
    }

    int j, zero_ind, num_zero_rows, zrow_ind;
    int i = 0;
    double del; 
    Matrix sketch, U, Vt;  
    sketch = zeros(l, mat->ncols);
    int* zero_rows = malloc(sketch.nrows * sizeof(int));
    // svals will be the singular values of the sketch matrix
    int svals_len = (int)fmin(mat->ncols, l);
    // need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: should this be here or only in svd 
    // REDUCED SVD calls for l X mat->ncols
    Vt = zeros(svals_len, sketch.ncols);
    // U in the svd of sketch (will always be l rows) 
    U = zeros(sketch.nrows, svals_len);
    // diag matrix used to hold 
    // run while we have rows left to sketch 
    num_zero_rows = zero_row_2(&sketch, zero_rows);
    zrow_ind = 0;
    // what if we change this to same as sketch.py, for loop
    for (i=0; i < mat->nrows; i++)
    {
        // insert rows of mat into the zero rows of sketch
        if (zrow_ind < num_zero_rows)
        {
            // insert rows of mat into the zero rows of sketch
            for (j=0; j<mat->ncols; j++)
                set_ind(&sketch, zero_rows[zrow_ind], j, get_ind(mat, i, j));
            zrow_ind++;
        }

        if (zrow_ind == num_zero_rows)
        {
            // no more zero rows 
            // run sketching procedure 
            // using reduced SVD
            if(svd_l(&sketch, &U, svals, &Vt, true) == 0)
            {
                // should pick out the middle singular value
                del = svals[(int)floor(svals_len/2)];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=0; j<svals_len; j++)
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                // new sketch is EV^T
                mult_diag(&sketch, svals, svals_len, &Vt);
            }
            else
            {
                printf("SVD Failed\n");
                exit(1);
            }
            // update zero rows 
            num_zero_rows = zero_row_2(&sketch, zero_rows); 
            zrow_ind = 0;
        }
    }
    // TODO: clean up everything 
    free(svals);
    free(zero_rows);
    free_mat(&Vt);
    free_mat(&U);
    return sketch;
}

// can't just change the svd to a pointer based on a flag cause it returns Vt
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
    // decide which svd implementation we are going to use
    

    int j, zero_ind, num_zero_rows, zrow_ind;
    int i = 0;
    double del; 
    Matrix sketch, V, V_trunc, V_trunc_t, E;  
    sketch = zeros(l, mat->ncols);

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
        //TODO: do we need to zero out svals, U, V each time? dont think so 
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
            // TODO: does this matter            
            if (svd(&sketch, svals, &V, true) == 0)
            {
                V_trunc = truncate_cols_2(&V, svals_len);
                // should pick out the middle singular value
                del = svals[(int)floor(svals_len/2)];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=0; j<svals_len; j++)
                {
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
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
    }
    // REPEAT ABOVE LOGIC to sketch
    // so why is our program so brutal? 
    // shoud we only do this if we don't have any zero rows at the end?
    zero_ind = zero_row(&sketch);
    if (zero_ind == -1)
    {
        if (svd(&sketch, svals, &V, false) == 0)
        {
            // do we need to truncate the columns of V? Yes for now 
            V_trunc = truncate_cols_2(&V, svals_len);
            // should pick out the middle singular value
            del = svals[(int)floor(l/2)];
            // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
            for(j=0; j<svals_len; j++)
            {
                svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
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

//construct using lapack sketch 
int construct_sketch_l(char* fname, int l, bool write, char* write_fname, bool check)
{
    Matrix mat = read_mat(fname);
    //struct timeval begin, end;
    clock_t begin, end;
    double time_spent;  
    //gettimeofday(&begin, NULL);
    begin = clock();
    Matrix sketch = fd_sketch_l(&mat, l);
    end = clock();
    //gettimeofday(&end, NULL);
    time_spent = (double) (end - begin)/ CLOCKS_PER_SEC;
    //time_spent = (double) (end.tv_sec - begin.tv_sec) * 1000.0;
    //printf("%f\n", time_spent);
    if (write)
        write_mat(&sketch, write_fname);
    if (check)
    {
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
    return 0;
}

// make sure that we satisfy the bound on reconstruction of covariance matrix
int test_sketch(char* fname, int l, bool write, char* write_fname)
{
    Matrix mat = read_mat(fname);
    // run time code here 
    Matrix sketch = fd_sketch(&mat, l);
    //
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
// TODO: should we give the option of OG svd?? dont want to 
int main(int argc, char* argv[])
{
    clock_t begin, end;
    double time_spent;  
    //gettimeofday(&begin, NULL);
    begin = clock();

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
    bool check = false; 
    if (construct_sketch_l(fname, l, wflag, write_fname, check) == 0)
    {
        end = clock();
        time_spent = (double) (end - begin)/ CLOCKS_PER_SEC;
        printf("%f ", time_spent);
        return 0;
    }
    else
        return 1;    
}
