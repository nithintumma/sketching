#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

//TODO: DO WE NEED TO FREE THE ARRAY INSIDE STRUCT

// test loat equality 
#define FLOAT_EQ(a, b)      (fabs(a - b) < 0.00001)

// matrix struct
typedef struct 
{
    unsigned int ncols;
    unsigned int nrows;
    // does this need to be a pointer to a pointer? 
    float* matrix;
} Matrix;

Matrix init_mat(int rows, int cols)
{
    //TODO: REMOVE
    Matrix mat;
    mat.nrows = rows;
    mat.ncols = cols;
    mat.matrix = (float*)malloc(rows*cols*sizeof(float));
    return mat;
}

// compute array index given r, c
// row-wise stored
float get_ind(Matrix* mat, int x, int y)
{
    // return float at row x, col y of matrix 
    return mat->matrix[x * mat->nrows + y];
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
        mat.matrix[i] = (float)i;
    return mat; 
}

// creates a matrix with random entries in [0, 1]
Matrix rand_matrix(int rows, int cols)
{
    srand(time(NULL));
    Matrix mat = init_mat(rows, cols);
    int n = mat.nrows * mat.ncols;
    float rand_num; 
    for (unsigned int i=0; i<n; i++)
    {
        rand_num = (float)rand()/(float)(RAND_MAX);
        printf("%0.2f\n", rand_num); 
        mat.matrix[i] = rand_num;
    }
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

// Need a function to print matrix
void print_mat(Matrix* mat)
{
    printf("%d X %d Matrix\n", mat->nrows, mat->ncols);
    for (int x=0; x < mat->nrows; x++)
    {
        for (int y=0; y < mat->ncols; y++)
        {
            printf(" %0.2f ", get_ind(mat, x, y)); 
        }
        printf("\n");
    }
}

// Need a function to write matrix to file 
void write_mat(Matrix* mat, char* fname)
{
    // write as txt
    FILE* fp = fopen(fname, "w"); 
    fprintf(fp, "%d %d\n", mat->nrows, mat->ncols);
    for (int x=0; x < mat->nrows; x++)
    {
        for (int y=0; y < mat->ncols; y++)
        {
            fprintf(fp, " %0.2f ", get_ind(mat, x, y)); 
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
    // buffer to hold line
    char* buf = malloc(1024 * sizeof(char));
    if (fgets(buf, 1024, fp) != NULL)
    {
        sscanf(buf, "%d %d", &nrows, &ncols);
    }
    int n = nrows * ncols; 
    // create the array that will store the Matrix
    Matrix mat = init_mat(nrows, ncols);
    // read the file line by line
    int i = 0;
    int bytes_read = 0;
    int total_bytes_read = 0;
    //char* line = (char) malloc(1024 * sizeof(char));
    while(fgets(buf, 1024, fp) != NULL)
    {
        // how do we read in the values from the current line? 
        for (int j=0; j<ncols; j++)
        {
            sscanf(buf + total_bytes_read, "%f%n", &mat.matrix[i*nrows + j], &bytes_read);
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

// add two matrices, return new matrix
Matrix add(Matrix* mat1, Matrix* mat2)
{
    if (mat1->nrows != mat2->nrows && mat1->ncols != mat2->ncols)
    {
        printf("Cannot add matrices without the same shape");
        exit(1);
    }
    Matrix result = init_mat(mat1->nrows, mat1->ncols);
    int n = result.nrows * result.ncols;
    for (int i = 0; i < n; i++)
        result.matrix[i] = mat1->matrix[i] + mat2->matrix[i];
    return result;
}

// returns multiply the matrix by the scalar c 
Matrix mult_scalar(Matrix* mat, float c)
{
    Matrix result = init_mat(mat->nrows, mat->ncols);
    int n = result.nrows * result.ncols;
    for (int i = 0; i < n; i++)
        result.matrix[i] = c * mat->matrix[i];
    return result;
}

// return c X n matrix transpose of input
// TODO: FIX THIS FUNCTION 
Matrix transpose(Matrix* mat)
{
    Matrix result = init_mat(mat->ncols, mat->nrows);
    // iterate through the rows cols of the original matrix 
    for (int i=0; i<mat->nrows; i++)
        for (int j=0; j<mat->ncols; j++)
            result.matrix[j*result.nrows + i] = mat->matrix[i*mat->nrows + j];
    return result;
}

// tests if matrices are elementwise equal using float equality check 
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

// TODO: MATH FUNCTIONS
// mutliply two matrices
// calculate matrix norms? Frobenius
// compute SVD 

void test_read_write()
{
    // create random matrix, print it
    // write it, read it, and print again (verify if equal) 
    int r = 10;
    int c = 10; 
    Matrix random = rand_matrix(r, c);
    print_mat(&random);
    write_mat(&random, "out_mat.txt");
    Matrix random_mat_2  = read_mat("out_mat.txt");
    print_mat(&random_mat_2);
}

void test_add(int r)
{
    // create two identity matrices, add them, and prini it
    Matrix mat1 = eye(r);
    Matrix mat2 = eye(r);
    Matrix result = add(&mat1, &mat2);
    print_mat(&result);
}

void test_transpose(int r, int c)
{
    Matrix mat = rand_matrix(r, c);
    print_mat(&mat);
    Matrix mat_t = transpose(&mat);
    print_mat(&mat_t);
}

void test_equal(int r, int c)
{
    Matrix mat = rand_matrix(r, c);
    Matrix mat1 = eye(r);
    Matrix mat2 = eye(r);
    if (equal(&mat, &mat) && equal(&mat1, &mat2))
        printf("Passed\n");
    else
        printf("Failed\n");
}

// test what we have so far
int main(int argc, char* argv[])
{
    // create a zero matrix 
    // what do we do here? 
    int r = 5;
    int c = 3;
    Matrix mat = rand_matrix(r, c);
    print_mat(&mat);
}

