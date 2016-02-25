#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
// gives us access to clapack
#include <Accelerate/Accelerate.h>
#include "matrix.h"

#define FLOAT_EQ(a, b) (fabs(a - b) < 0.000001)
#define IMIN(a, b)  ((a < b) ? (a) : (b))

// define GET_IND, SET_IND for column major format 
// a is a double* pointer to top left of array 
#define GET(a, r, c, i, j) (a[j * r + i])
#define SET(a, r, c, i, j, val) (a[j*r + i] = val)
//TODO: might not need these since we are using two-dim C arrays 

// need the dgsdd prototype
extern void dgesdd( char* jobz, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* iwork, int* info );
void print_array(double* m, int r, int c);

/*
explains leading dims: 
http://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order

params to dgesdd
char* jobz - type of SVD to perform, 'A' for full 
int* m - rows
int* n - cols
double* a - matrix, m X n? TODO: stored in column major order tho
            jobs='O' - a is overwriten with the first n columns of U, unless m >= n, when Vt
                        is             
            jobsi !-= 'O' - destroyed 
int* lda - leading dimension of a LDA >= max(1,Mb)
double* s - output, singular values (should be n long?)
double* u - output, UCOL=M 
int* ldu - leading dim of U
double* vt - output, vT (transpose) 
int* ldvt -leading dimension of vt
double* work - on exit, work(1) returns optimal work
int* lwork - dimension of work, should just run once and let it calculate opt
int* iwork - dimension: 8 * IMIN(n, m)
int* info - 
*/
// switch row to column order, in place (transpose)
// in here we could always just do performant implementations
// probably better long term, can always just copy to matrix struct at end
void row_to_col(Matrix *m, double* mC)
{
    // mT should be m_cols X m_rows multidim array 
    // how do we know that mC is properly alloced? 
    for(int i=0; i< m->nrows; i++)
    {
        for (int j=0; j< m->ncols; j++)
            // get_ind accesses row order, SET accesses column order
            SET(mC, m->nrows, m->ncols, i, j, get_ind(m, i, j));
    }
    // mT will be COL major version of input m
}

// really want to return a row major matrix 
void col_to_row(double* m, int r, int c, Matrix* mR)
{
    // m is column major, want to return mT which is row major
    // mR should be initied to r X c matrix
    if ((mR->nrows != r) || (mR->ncols != c))
    {
        printf("Cannot convert matrices that have different shapes\n");
        exit(1);
    }
    for(int i=0; i< r; i++)
    {
        for (int j=0; j< c; j++)
            // get_ind accesses row order, SET accesses column order
            set_ind(mR, i, j, GET(m, r, c, i, j));
    }
    // on return, mR should have contents of m stored in row order
}

int svd_l(Matrix* mat, Matrix* Umat, double* s, Matrix* VTmat, bool reduced)
{
    // mat should be nrows X ncosl
    // U should be nrows X nrows
    // s should be MIN(nrows, ncols) - TODO: caller allocates ncols always!!
    // V should be ncols X ncols
    // reduced should be false
    int mval = mat->nrows;
    int *m = &mval;
    int nval = mat->ncols;
    int *n = &nval;
    int *lda = m;
    char job; 
    int ldvt_val, *ldvt, *ldu; 
    int vcols, ucols; 
    if (reduced)
    {
        // TODO, check that Umat, VTmat have right dims? 
        job = 'S';
        ucols =  IMIN(*m, *n);
        ldvt_val = IMIN(*m, *n);
        ldvt = &ldvt_val;
    }
    else
    {
        job = 'A';
        ucols = *m;
        double *Vt[*n][*n];
        ldvt = n;
    }
    ldu = m; 
    vcols = *n; 
    // we want to reuse the input matrix arrays 
    double *A, *U, *Vt;
    if (STORE == 1)
    {
        // matrix sturcts are COL major, no conversion necessary
        A = mat->matrix;
        U = Umat->matrix;
        Vt = VTmat->matrix;
    }
    else 
    {
        // matrix structs are row major, need to allocate new arrays and convert at end
        A = malloc(*m * *n * sizeof(double));
        row_to_col(mat, A);
        U = malloc(*m * ucols * sizeof(double));
        Vt = malloc(*ldvt * vcols * sizeof(double));
    }
    char *jobz = &job;
    int *iwork = malloc(8 * IMIN(*m, *n) * sizeof(int));
    int info_val, *info;
    info_val = 1; 
    info = &info_val;
    // call dgesdd once with lwork=-1 to get size of work  
    double opt_work[1];
    int lwork_val = -1;
    int *lwork = &lwork_val;
    dgesdd_(jobz, m, n, A, lda, s, U, ldu, Vt, ldvt, 
            opt_work, lwork, iwork, info);
    // opt_work contains the best size for work
    lwork_val = (int) opt_work[0];
    lwork = &lwork_val;
    double* work = malloc(lwork_val * sizeof(double));
    dgesdd_(jobz, m, n, A, lda, s, U, ldu, Vt, ldvt, 
            work, lwork, iwork, info);
    if (STORE == 0)
    {
        // matrix structs are ROW major 
        col_to_row(U, *ldu, ucols, Umat);
        col_to_row(Vt, *ldvt, vcols, VTmat);
    }
    // TODO: note that we are returning Vt, not V here 
    // TODO: check error from svd, return 1 if failed 
    return 0; 
}

// TODO: don't be lazy and do the highlight thing, can reuse bold,reverse

void print_array(double* a, int r, int c)
{
    // assume that double* a is in column order 
    double val; 
    for (int i=0; i<r; i++)
    {
        for (int j=0; j<c; j++)
        {
            val = GET(a, r, c, i, j);
            if (val < 0)
                printf("%0.2f ", val);
            else
                printf(" %0.2f ", val);
        }
        printf("\n");
    }
}

int run_sketch_l()
{
    // what should we do to test it? 
    char* mat_fname = "test_matrices/small_svd_mat.txt";
    Matrix mat = read_mat(mat_fname);
    // call svd and see what happens!
    Matrix V = init_mat(mat.ncols, mat.ncols);
    Matrix U = init_mat(mat.nrows, mat.nrows);
    // this might want to be MAX(nrows, ncols) 
    double *w = malloc(mat.ncols * sizeof(double));
    bool reduced = false;
    int err = svd_l(&mat, &U, w, &V, reduced);
    if (err != 0)
        return 1;
    else 
        return 0;
}
