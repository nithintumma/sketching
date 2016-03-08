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
#define IMAX(a, b) ((a > b) ? a : b)

//TODO: all of these sketching functions use the same type of code
// make them all share code, so we don't have to rewrite it each time
// the abstraction can be l, batch size, del_ind, start_row, end_row, 
// fd: l = l/2, batch_size = l/2, del_ind = l/2, alpha = 1.0
// alpha: l = l/2, batch_size = l/2, del_ind = l/2, alpha = alpha
// batched: l = l, batch_size = b, del_ind = l-1, alpha = 1.0
// batched alpha: l = l, batch_size = b, del_ind = l-1, alpha = alpha

//TODO: change the order of for loops around for col order?
int get_zero_rows(Matrix* mat, int* zero_rows)
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

// returns the sketch of the input matrix, processes it one row at a time 
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

// once we have batched-FD
Matrix batch_fd_sketch_l(Matrix* mat, int l, int batch_size)
{
    // do checks 
    if (l >= mat->nrows)
    {
        printf("Cannot sketch into more rows than input\n");
        exit(1);
    } 
    // basically, we will add batch_size rows to sketch, do SVD
    // truncate, repeat 
    // sketch will be (l + batch_size) X ncols 
        // init everything 
    int i, j, k, batch_start_row, zero_ind, num_zero_rows, zrow_ind; 
    double del; 
    Matrix sketch, U, Vt;  
    sketch = zeros((l + batch_size), mat->ncols);
    int* zero_rows = malloc(sketch.nrows * sizeof(int));

    // svals will be the singular values of the sketch matrix
    // we will only get min(mat->ncols, l+batch_size) svals, but we need (l+batch_size) for mult
    //(int)fmin(mat->ncols, l+batch_size);
    int svals_len = IMAX(sketch.nrows, sketch.ncols);
    if (l < svals_len)
    {
        printf("Cannot have l < mat->ncols\n");
        exit(1);
    }
    // need to allocate for ncols even though we only use the first l 
    // zero out last ones 
    double* svals = malloc(sizeof(double) * svals_len);
    for (int i=0; i<svals_len; i++)
        svals[i] = 0.0;

    // each time, we take the l'th singular value to subtract 
    int del_ind = l - 1;

    // REDUCED SVD calls for l X mat->ncols
    // we want E x Vt = sketch 
    Vt = zeros(svals_len, sketch.ncols);
    // U in the svd of sketch (will always be l rows) 
    U = zeros(sketch.nrows, svals_len);
    num_zero_rows = get_zero_rows(&sketch, zero_rows);
    zrow_ind = 0;
    // add l rows to the sketch to start off 
    for (i=0; i<l; i++)
    {
        for(j=0; j<mat->ncols; j++)
            set_ind(&sketch, i, j, get_ind(mat, i, j));
    }
    // each time, we want to add a batch_size rows to the sketch 
    // IF THIS DOESN"T WORK, WE CAN GO BACK to 0's, but that seems unnecessary
    for (i=l; i<mat->nrows/batch_size; i++)
    {
        // add b rows to the sketch 
        batch_start_row = i * batch_size;
        for(k=0; k<batch_size; k++)
        {
            // first (l-1) rows of sketch should be non-zero 
            for (j=0; j<mat->ncols; j++)
                set_ind(&sketch, l-1+k, j, get_ind(mat, batch_start_row + k, j));
        }
        // sketch it
        if(svd_l(&sketch, &U, svals, &Vt, true) == 0)
        {
            // make bigger, zero out rows for mult
            // should pick out the middle singular value
            del = SQUARE(svals[del_ind]);
            // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
            for(j=0; j<del_ind; j++)
                svals[j] = sqrt(fmax(SQUARE(svals[j]) - del, 0.0));
            for (j=del_ind; j<svals_len; j++)
                svals[j] = 0.0;
            // new sketch is EV^T
            mult_diag(&sketch, svals, svals_len, &Vt);
            // we need to print out sketch to make sure that the last b rows are zero 
        }
        else
        {
            printf("SVD Failed\n");
            exit(1);
        }
        // update zero rows 
        num_zero_rows = get_zero_rows(&sketch, zero_rows); 
        zrow_ind = 0;
    }
    // TODO: clean up everything 
    free(svals);
    free(zero_rows);
    free_mat(&Vt);
    free_mat(&U);
    return sketch; 
}

// what is the code for changing the sketch size as we are processing the stream? 
// maybe we should do all of this code in Python if we don't care about runtime  

// sketch only uses l/2 rows 
// Running fast version, explained thoroughly in http://content.lib.utah.edu/utils/getfile/collection/etd3/id/3327/filename/3340.pdf
Matrix pfd_sketch_l(Matrix* mat, int l, float alpha)
{
    if (!((alpha >=0) && (alpha <=1)))
    {
        printf("Alpha needs to be in [0, 1]\n");
        exit(1);
    }
    // would be nice if l was a multiple of 2 as well
    if (!(((l % 2) == 0) && l > 2 && (l < mat->ncols) && (l < mat->nrows)))
    {
        printf("Require sketch size to be a multiple of 2, >= 4, < ncols %d\n", l);
        exit(1);
    }
    // init everything 
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
    num_zero_rows = get_zero_rows(&sketch, zero_rows);
    zrow_ind = 0;
    // index of the first singular value to be reduced
    // make sure we don't go too far 
    int t = (int) floor((alpha * l)/2.0);
    // do we need to subtract 1?
    int alpha_ind = (int) fmax(l - 2 * t, 0);
    if (alpha_ind >= svals_len)
        alpha_ind = svals_len - 1;

    // DO WE NEED TO SUBTRACT 1?? 
    int del_ind = (int) fmax(l - t, 0);
    //printf("T: %d, Alpha ind: %d, Del ind: %d\n", t, alpha_ind, del_ind);
    
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
                del = svals[del_ind];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=alpha_ind; j<svals_len; j++)
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
            num_zero_rows = get_zero_rows(&sketch, zero_rows); 
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

void update_fd_sketch_l(Matrix* sketch, Matrix* mat, int l)
{
    // add the wors form mat to the sketch and sketch, 
    // hope that the sketch has sufficient zero rows, 
    if (l >= sketch->ncols)
    {
        printf("Cannot sketch into more rows than columns\n");
        exit(1);
    }
    if (l != sketch->nrows)
    {
        printf("l should be the same number of rows as the input sketch\n");
        exit(1);
    }
    if (sketch->ncols != mat->ncols)
    {
        printf("Sketch and input matrix need to have compatible shapes\n");
        exit(1);
    }
    // what do I do here?
    int i, j, zero_ind, num_zero_rows, zrow_ind;
    double del; 
    Matrix U, Vt;  
    int* zero_rows = malloc(sketch->nrows * sizeof(int));
    // svals will be the singular values of the sketch matrix
    int svals_len = (int)fmin(mat->ncols, l);
    int del_ind = (int)floor(svals_len/2);
    // need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: should this be here or only in svd 
    // REDUCED SVD calls for l X mat->ncols
    Vt = zeros(svals_len, sketch->ncols);
    // U in the svd of sketch (will always be l rows) 
    U = zeros(sketch->nrows, svals_len);
    // run while we have rows left to sketch 
    num_zero_rows = get_zero_rows(sketch, zero_rows);
    zrow_ind = 0;
    for (i=0; i < mat->nrows; i++)
    {
        // insert rows of mat into the zero rows of sketch
        if (zrow_ind < num_zero_rows)
        {
            // insert rows of mat into the zero rows of sketch
            for (j=0; j<mat->ncols; j++)
                set_ind(sketch, zero_rows[zrow_ind], j, get_ind(mat, i, j));
            zrow_ind++;
        }

        if (zrow_ind == num_zero_rows)
        {
            // no more zero rows 
            // run sketching procedure 
            // using reduced SVD
            if(svd_l(sketch, &U, svals, &Vt, true) == 0)
            {
                // should pick out the middle singular value
                del = svals[del_ind];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=0; j<svals_len; j++)
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                // new sketch is EV^T
                mult_diag(sketch, svals, svals_len, &Vt);
            }
            else
            {
                printf("SVD Failed\n");
                exit(1);
            }
            // update zero rows 
            num_zero_rows = get_zero_rows(sketch, zero_rows); 
            zrow_ind = 0;
        }
    }
    // TODO: clean up everything 
    free(svals);
    free(zero_rows);
    free_mat(&Vt);
    free_mat(&U);
    // don't return anything, we should have modified the sketch in place 
}


Matrix sub_fd_sketch_l(Matrix* mat, int start_row, int end_row, int l)
{
    int num_rows = end_row - start_row + 1;
    if (num_rows <= 0)
    {
        printf("Need to sketch a positive number of rows\n");
        exit(1);
    }

    if (l >= num_rows)
    {
        printf("Cannot sketch into more rows than input: %d %d\n", l, num_rows);
        exit(1);
    }

    // would be nice if l was a multiple of 2 as well
    if (!(((l % 2) == 0) && l > 2 && (l < mat->ncols)))
    {
        printf("Require sketch size to be a multiple of 2, >= 4, < ncols %d\n", l);
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
    int del_ind = (int)floor(svals_len/2);
    // need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: should this be here or only in svd 
    // REDUCED SVD calls for l X mat->ncols
    Vt = zeros(svals_len, sketch.ncols);
    // U in the svd of sketch (will always be l rows) 
    U = zeros(sketch.nrows, svals_len);
    // run while we have rows left to sketch 
    num_zero_rows = get_zero_rows(&sketch, zero_rows);
    zrow_ind = 0;
    for (i=start_row; i <= end_row; i++)
    {
        // insert rows of mat into the zero rows of sketch
        if (zrow_ind < num_zero_rows)
        {
            // insert row of mat into the zero rows of sketch
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
                del = svals[del_ind];
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
            num_zero_rows = get_zero_rows(&sketch, zero_rows); 
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
    if (!(((l % 2) == 0) && l > 2 && (l < mat->ncols)))
    {
        printf("Require sketch size to be a multiple of 2, >= 4, < ncols %d\n", l);
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
    int del_ind = (int)floor(svals_len/2);
    // need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: should this be here or only in svd 
    // REDUCED SVD calls for l X mat->ncols
    Vt = zeros(svals_len, sketch.ncols);
    // U in the svd of sketch (will always be l rows) 
    U = zeros(sketch.nrows, svals_len);
    // run while we have rows left to sketch 
    num_zero_rows = get_zero_rows(&sketch, zero_rows);
    zrow_ind = 0;
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
                del = svals[del_ind];
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
            num_zero_rows = get_zero_rows(&sketch, zero_rows); 
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

//construct using lapack sketch 
int construct_sketch_l(char* fname, int l, float alpha, bool write, char* write_fname, bool check)
{
    Matrix mat = read_mat(fname);
    //struct timeval begin, end;
    clock_t begin, end;
    double time_spent;  
    //gettimeofday(&begin, NULL);
    begin = clock();
    //Matrix sketch = fd_sketch_l(&mat, l);
    Matrix sketch = pfd_sketch_l(&mat, l, alpha);
    end = clock();
    //gettimeofday(&end, NULL);
    time_spent = (double) (end - begin)/ CLOCKS_PER_SEC;
    //time_spent = (double) (end.tv_sec - begin.tv_sec) * 1000.0;
    //printf("%f\n", time_spent);
    if (write)
        write_mat(&sketch, write_fname);
    // don't call this 
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
