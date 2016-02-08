#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

//TODO: DO WE NEED TO FREE THE ARRAY INSIDE STRUCT
//TODO: ADD A SETTER FUNCTION FOR MATRIX INDEXING 

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
    // return float at row x, col y of matrix (0 indexed)
    return mat->matrix[x * mat->ncols + y];
}

// set the matrix at x, y to val
void set_ind(Matrix* mat, int x, int y, float val)
{
    mat->matrix[x * mat->ncols + y] = val;
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
        mat.matrix[i] = (float)rand()/(float)(RAND_MAX);
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
    fprintf(fp, "# %d %d\n", mat->nrows, mat->ncols);
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
    int buf_size = 4096;
    char* buf = malloc(buf_size * sizeof(char));
    if (fgets(buf, buf_size, fp) != NULL)
        sscanf(buf, "# %d %d", &nrows, &ncols);
    int n = nrows * ncols; 
    // create the array that will store the Matrix
    Matrix mat = init_mat(nrows, ncols);
    // read the file line by line
    int i = 0;
    int bytes_read = 0;
    int total_bytes_read = 0;
    // maybe check if we have read more lines than we need? 
    while(fgets(buf, buf_size, fp) != NULL)
    {
        // how do we read in the values from the current line? 
        for (int j=0; j<ncols; j++)
        {
            sscanf(buf + total_bytes_read, "%f%n", &mat.matrix[i*ncols + j], &bytes_read);
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
Matrix transpose(Matrix* mat)
{
    Matrix result = init_mat(mat->ncols, mat->nrows);
    // iterate through the rows cols of the original matrix 
    for (int i=0; i<mat->nrows; i++)
        for (int j=0; j<mat->ncols; j++)
            result.matrix[j*result.ncols + i] = mat->matrix[i*mat->ncols + j];
    return result;
}

// returns new matrix that is the matrix product of the inputs (mat1 * mat2)
Matrix mult(Matrix* mat1, Matrix* mat2)
{
    if (mat1->ncols != mat2->nrows)
    {
        printf("Cannot multiply matrices, incompatible shapes");
        exit(1);
    }
    // look up the matrix multiplication function 
    float sum;
    Matrix result = init_mat(mat1->nrows, mat2->ncols);
    for(int i=0; i<mat1->nrows; i++)
    {
        for(int j=0; j<mat2->ncols; j++)
        {
            for(int k=0; k<mat2->nrows; k++)
            {
                sum += get_ind(mat1, i, k) * get_ind(mat2, k, j);
            }
            result.matrix[i*result.ncols + j] = sum;
            sum = 0.0;
        }
    }
    return result;
}

// returns frobenius norm of the matrix
float frobenius_norm(Matrix* mat)
{
    float sum;
    for (int i=0; i<mat->ncols * mat->nrows; i++)
        sum += mat->matrix[i] * mat->matrix[i];
    return sqrt(sum);
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
// compute SVD 

void old_test_mult(int r)
{
    Matrix mat1 = eye(r);
    Matrix mat2 = mult_scalar(&mat1, 3.0);
    Matrix mat3 = mult_scalar(&mat1, 4.0);
    Matrix result = mult(&mat2, &mat3);
    print_mat(&result);
}

void test_frob_norm(int r, int c)
{
    Matrix mat = eye(r);
    float norm = frobenius_norm(&mat);
    printf("%f Norm\n", norm);
}

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

void old_test_add(int r)
{
    // create two identity matrices, add them, and prini it
    Matrix mat1 = eye(r);
    Matrix mat2 = eye(r);
    Matrix result = add(&mat1, &mat2);
    print_mat(&result);
}

void old_test_transpose(int r, int c)
{
    Matrix mat = rand_matrix(r, c);
    print_mat(&mat);
    Matrix mat_t = transpose(&mat);
    print_mat(&mat_t);
}

void old_test_equal(int r, int c)
{
    Matrix mat = rand_matrix(r, c);
    Matrix mat1 = eye(r);
    Matrix mat2 = eye(r);
    if (equal(&mat, &mat) && equal(&mat1, &mat2))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void old_test_mult_scalar(int r, int c)
{
    Matrix mat = rand_matrix(r, c);
    print_mat(&mat);
    Matrix mat2 = mult_scalar(&mat, 2.0);
    print_mat(&mat2);
}

void test_add()
{
    printf("Testing Add\n");
    char* mat1_fname = "test_matrices/mat1.txt";
    char* mat2_fname = "test_matrices/mat3.txt";
    char* result_fname = "test_matrices/sum_mat1_3.txt";
    Matrix mat1 = read_mat(mat1_fname);  
    Matrix mat2 = read_mat(mat2_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = add(&mat1, &mat2);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void test_transpose()
{
    printf("Testing Transpose\n"); 
    char* mat1_fname = "test_matrices/mat1.txt";
    char* result_fname = "test_matrices/mat1_t.txt";
    Matrix mat1 = read_mat(mat1_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = transpose(&mat1); 
    printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void test_scalar_mult()
{
    printf("Testing Scalar Mult\n");
    char* mat1_fname = "test_matrices/mat1.txt";
    Matrix mat1 = read_mat(mat1_fname);
    Matrix true_result = read_mat(result_fname);
    char* result_fname = "test_matrices/scal_mult_mat1_3.2.txt";
    Matrix result = mult_scalar(&mat1, 3.2)
    printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

//TODO: FIX THIS 
void test_mult()
{
    printf("Testing Matrix Mult\n");
    char* mat1_fname = "test_matrices/mat1.txt";
    char* mat2_fname = "test_matrices/mat2.txt";
    char* result_fname = "test_matrices/mult_mat1_2.txt";
    Matrix mat1 = read_mat(mat1_fname);  
    Matrix mat2 = read_mat(mat2_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = mult(&mat1, &mat2);
    printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

// test what we have so far
int main(int argc, char* argv[])
{
    test_add();
    test_transpose();
    test_mult();
}

