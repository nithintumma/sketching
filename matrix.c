#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"

//TODO: DO WE NEED TO FREE THE ARRAY INSIDE STRUCT
//TODO: USE SET_IND EVERYWHERE SO WE CAN CHANGE ROW/COLUMN


// test double equality 
#define FLOAT_EQ(a, b)      (fabs(a - b) < 0.0001)

// NOT SURE IF THIS IS THE RIGHT WAY TO DO THIS!!
#if (STORE == 0)
    inline double get_ind(Matrix* mat, int x, int y)
    {
        return mat->matrix[x * mat->ncols + y];
    }
    inline void set_ind(Matrix* mat, int x, int y, double val)
    {
        mat->matrix[x * mat->ncols + y] = val;
    }
#else 
    inline double get_ind(Matrix* mat, int x, int y)
    {
        return mat->matrix[y * mat->nrows + x];
    }
    inline void set_ind(Matrix* mat, int x, int y, double val)
    {
        mat->matrix[y * mat->nrows + x] = val;
    }
#endif

Matrix init_mat(int rows, int cols)
{
    Matrix mat;
    mat.nrows = rows;
    mat.ncols = cols;
    mat.matrix = (double*)malloc(rows*cols*sizeof(double));
    return mat;
}

void free_mat(Matrix *mat)
{
    // free the matrix
    free(mat->matrix);
}

// do multiplication in place 
void mult_diag(Matrix* result, double* diag, int diag_len, Matrix* mat2)
{
    // return the result of diagonal(diag) * mat2
    // assert that the result is the right shape 
    if((result->nrows != diag_len) || (mat2->nrows != diag_len) || (result->ncols != mat2->ncols))
    {
        printf("Matrices not of the right shape\n");
        exit(1);
    }
    double val; 
    // this is not the order that we want to run the for loop in 
    // multiply the ith row of mat2 by diag[i]
    // DO THIS ORDER FOR EFFICIENCY 
    for (int j=0; j<mat2->ncols; j++)
        for (int i=0; i<mat2->nrows; i++)
        {
            val = diag[i] * get_ind(mat2, i, j);
            set_ind(result, i, j, val);
        }
}


// construct a diagonal matrix with input array as diagonal
Matrix diag(double* diagonal, int rows)
{
    Matrix mat = zeros(rows, rows);
    for (int i =0; i<rows; i++)
        set_ind(&mat, i, i, diagonal[i]);
    return mat;
}

// creates a matrix with given shape initalized to 0.0
Matrix zeros(int rows, int cols) 
{ 
    // creats a rows x cols matrix initialized to zeros 
    Matrix mat = init_mat(rows, cols);
    int n = rows * cols;
    for (unsigned int i=0; i<n; i++)
        mat.matrix[i] = 0.0;  
    return mat; 
}

Matrix test_ind(int rows, int cols)
{
    int n = rows * cols;
    Matrix mat = init_mat(rows, cols); 
    for (unsigned int i=0; i<n; i++)
        mat.matrix[i] = (double)i;
    return mat; 
}

// creates a matrix with random entries in [0, 1]
Matrix rand_matrix(int rows, int cols)
{
    srand(time(NULL));
    Matrix mat = init_mat(rows, cols);
    int n = mat.nrows * mat.ncols;
    double rand_num; 
    for (unsigned int i=0; i<n; i++)
        mat.matrix[i] = (double)rand()/(double)(RAND_MAX);
    return mat; 
}

// creates square identity matrix
Matrix eye(int rows)
{
    Matrix mat = zeros(rows, rows);
    for (int i = 0; i < rows; i++)
    {
        // Need to figure out indexing 
        mat.matrix[i * rows + i] = 1.0;
    }
    return mat; 
}

// does truncate matrix work? 
// trunacate Matrix with ncols < mat->ncols columns in place 
// what if we passed in a pointer to the pointer 
Matrix truncate_cols_2(Matrix* mat, int ncols)
{
    if (ncols > mat->ncols)
    {
        printf("Cannot truncate to more columns than original matrix\n");
        exit(1);
    }
    // maybe we should just return a pointer to the new matrix 
    static Matrix new_mat;
    new_mat = zeros(mat->nrows, ncols);
    static Matrix* new_mat_p;
    new_mat_p = &new_mat;

    for(int i=0; i<mat->nrows; i++)
    {
        for(int j=0; j<ncols; j++)
            set_ind(new_mat_p, i, j, get_ind(mat, i, j));
    }
    // what do I do here
    return new_mat;
}

// TODO: do this in place 
Matrix truncate_rows(Matrix* mat, int nrows)
{
    if (nrows > mat->nrows)
    {
        printf("Cannot truncate to more columns than original matrix\n");
        exit(1);
    }
    // maybe we should just return a pointer to the new matrix 
    static Matrix new_mat;
    new_mat = zeros(nrows, mat->ncols);
    static Matrix* new_mat_p;
    new_mat_p = &new_mat;

    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<mat->ncols; j++)
            set_ind(new_mat_p, i, j, get_ind(mat, i, j));
    }
    return new_mat;

}

// TODO: fix this function (don't think it works properly)
void truncate_cols(Matrix** mat, int ncols)
{
    if (ncols > (*mat)->ncols)
    {
        printf("Cannot truncate to more columns than original matrix\n");
        exit(1);
    }

    Matrix new_mat = init_mat((*mat)->nrows, ncols);


    //double* new_mat = (double*) malloc(mat->nrows * ncols * sizeof(double));
    // create new matrix
    // now what do we have to do 
    for (int i=0; i<(*mat)->nrows; i++)
    {
        for (int j=0; j<ncols; j++)
            //TODO: make this a setter function
            //new_mat[i * ncols + j] = get_ind(mat, i, j);
            set_ind(&new_mat, i, j, get_ind(*mat, i, j));
    }
    free_mat(*mat);
    *mat = &new_mat; 
    // clean up old matrix
    //free(mat->matrix);
    //mat->ncols = ncols;
    //mat->matrix = new_mat;
}

// Need a function to print matrix
void print_mat(Matrix* mat)
{
    printf("%d X %d Matrix\n", mat->nrows, mat->ncols);
    for (int x=0; x < mat->nrows; x++)
    {
        for (int y=0; y < mat->ncols; y++)
        {
            // TODO: deal with negatives taking up more space? 
            printf(" %0.2f ", get_ind(mat, x, y)); 
        }
        printf("\n");
    }
    printf("\n");
}

// Need a function to write matrix to file 
void write_mat(Matrix* mat, char* fname)
{
    // write as txt
    FILE* fp = fopen(fname, "w"); 
    fprintf(fp, "# %d %d\n", mat->nrows, mat->ncols);
    for (int x=0; x < mat->nrows; x++)
    {
        for (int y=0; y < mat->ncols; y++)
        {
            fprintf(fp, " %f ", get_ind(mat, x, y)); 
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Need a function to initialize a matrix from a file? 
Matrix read_mat(char* fname)
{
    FILE* fp = fopen(fname, "r"); 
    // read in the first line to get r, c 
    int nrows, ncols;
    // buffer to hold the header
    int h_buf_size = 1000; 
    char* h_buf = malloc(h_buf_size * sizeof(char));
    if (fgets(h_buf, h_buf_size, fp) != NULL)
        sscanf(h_buf, "# %d %d", &nrows, &ncols);
    int n = nrows * ncols; 
    free(h_buf);
    // buffer to hold line 
    // 24 for the largest # of chars in a float 
    int buf_size = ncols * 24;
    char* buf = malloc(buf_size * sizeof(char));

    // create the array that will store the Matrix
    Matrix mat = init_mat(nrows, ncols);
    // read the file line by line
    int i = 0;
    int bytes_read = 0;
    int total_bytes_read = 0;
    // maybe check if we have read more lines than we need? 
    double val;
    while(fgets(buf, buf_size, fp) != NULL)
    {
        // how do we read in the values from the current line? 
        for (int j=0; j<ncols; j++)
        {
            //TODO: replace this with set_ind if possible
            sscanf(buf + total_bytes_read, "%lf%n", &val, &bytes_read);
            //&mat.matrix[i*ncols + j]
            set_ind(&mat, i, j, val);
            total_bytes_read += bytes_read;
        }
        i++;
        total_bytes_read = 0;
        bytes_read = 0;
    }
    free(buf);
    fclose(fp);
    return mat; 
}
//TODO: change this to get/set?
// Assumes that the matrices are stored in the same way (row/col), should be fine
// add two matrices, return new matrix
Matrix add(Matrix* mat1, Matrix* mat2)
{
    if (mat1->nrows != mat2->nrows && mat1->ncols != mat2->ncols)
    {
        printf("Cannot add matrices without the same shape\n");
        exit(1);
    }
    Matrix result = init_mat(mat1->nrows, mat1->ncols);
    int n = result.nrows * result.ncols;
    for (int i = 0; i < n; i++)
        result.matrix[i] = mat1->matrix[i] + mat2->matrix[i];
    return result;
}

// subtract two matrices, return new matrix
Matrix subtract(Matrix* mat1, Matrix* mat2)
{
    if (mat1->nrows != mat2->nrows && mat1->ncols != mat2->ncols)
    {
        printf("Cannot subtract matrices without the same shape\n");
        exit(1);
    }
    Matrix result = init_mat(mat1->nrows, mat1->ncols);
    int n = result.nrows * result.ncols;
    for (int i=0; i<n; i++)
        result.matrix[i] = mat1->matrix[i] - mat2->matrix[i];
    return result;
}
// returns multiply the matrix by the scalar c 
Matrix mult_scalar(Matrix* mat, double c)
{
    Matrix result = init_mat(mat->nrows, mat->ncols);
    int n = result.nrows * result.ncols;
    for (int i = 0; i < n; i++)
        result.matrix[i] = c * mat->matrix[i];
    return result;
}

// return c X n matrix transpose of input
Matrix transpose(Matrix* mat)
{
    Matrix result = init_mat(mat->ncols, mat->nrows);
    // iterate through the rows cols of the original matrix 
    for (int i=0; i<mat->nrows; i++)
        for (int j=0; j<mat->ncols; j++)
            set_ind(&result, j, i, get_ind(mat, i, j));
    return result;
}

// returns new matrix that is the matrix product of the inputs (mat1 * mat2)
Matrix mult(Matrix* mat1, Matrix* mat2)
{
    if (mat1->ncols != mat2->nrows)
    {
        printf("Cannot multiply matrices, incompatible shapes\n");
        exit(1);
    }
    // look up the matrix multiplication function 
    double sum;
    Matrix result = init_mat(mat1->nrows, mat2->ncols);
    for(int i=0; i<mat1->nrows; i++)
    {
        for(int j=0; j<mat2->ncols; j++)
        {
            for(int k=0; k<mat2->nrows; k++)
            {
                sum += get_ind(mat1, i, k) * get_ind(mat2, k, j);
            }
            set_ind(&result, i, j, sum); 
            sum = 0.0;
        }
    }
    return result;
}

// returns squared frobenius norm of the matrix
double sq_frobenius_norm(Matrix* mat)
{
    double sum = 0.0;
    for (int i=0; i<mat->ncols * mat->nrows; i++)
        sum += mat->matrix[i] * mat->matrix[i];
    return sum;
}


// tests if matrices are elementwise equal using double equality check 
bool equal(Matrix* mat1, Matrix* mat2)
{
    if (mat1->nrows != mat2->nrows || mat1->ncols != mat2->ncols)
        return 0; 
    bool result = 1; 
    for (int i = 0; i<mat1->nrows * mat1->ncols; i++)
    {
        result &= FLOAT_EQ(mat1->matrix[i], mat2->matrix[i]);
    }
    return result; 
}

// Functions to interface with other SVD implementation
// return 2-d double array from matrix 
double** convert_mat(Matrix* mat)
{
    //double doub_mat[mat->nrows][mat->ncols];
    double** doub_mat = malloc(sizeof(double*) * mat->nrows);
    for (int i=0; i<mat->nrows; i++)
        doub_mat[i] = malloc(sizeof(double) * mat->ncols);

    for (int i=0; i<mat->nrows; i++)
        for (int j=0; j<mat->ncols; j++)
            doub_mat[i][j] = (double) get_ind(mat, i, j);
    return doub_mat;
}

// print matrix that is stored as two dimensional array (C row major)
void print_s_mat(double** mat, int nrows, int ncols)
{
    for (int i=0; i<nrows; i++)
    {
        for (int j=0; j<ncols; j++)
            printf("%0.2f ", mat[i][j]);
        printf("\n");
    }
}

