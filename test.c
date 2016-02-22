#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"

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
    float norm = sq_frobenius_norm(&mat);
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
    printf("Testing Add: ");
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
    printf("Testing Transpose: "); 
    char* mat1_fname = "test_matrices/mat1.txt";
    char* result_fname = "test_matrices/mat1_t.txt";
    Matrix mat1 = read_mat(mat1_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = transpose(&mat1); 
    // printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void test_truncate_cols()
{
    printf("Testing truncate_cols (visually): \n");
    char* mat_name = "test_matrices/small_mat.txt";
    Matrix mat = read_mat(mat_name);
    printf("Original matrix\n");
    print_mat(&mat);
    truncate_cols(&mat, mat.ncols-3);
    printf("Truncated matrix\n");
    print_mat(&mat);
    /*
    bool flag = true;
    if (trunc_mat.ncols == (mat.ncols - 3))
    {
        for (int i=0; i<mat.nrows; i++)
        {
            for (int j=0; j<trunc_mat.ncols; j++)
            {
                if (get_ind(&mat, i, j) != get_ind(&trunc_mat, i, j))
                    flag = false;
            }
        }
        if (flag)
            printf("Passed\n");
        else
            printf("Failed\n");
    }
    else
        printf("Failed\n");
    */
}


void test_scalar_mult()
{
    printf("Testing Scalar Mult: ");
    char* mat1_fname = "test_matrices/mat1.txt";
    char* result_fname = "test_matrices/scal_mult_mat1_3.2.txt";
    Matrix mat1 = read_mat(mat1_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = mult_scalar(&mat1, 3.2);
    //printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void test_mult()
{
    printf("Testing Matrix Mult: ");
    char* mat1_fname = "test_matrices/mat1.txt";
    char* mat2_fname = "test_matrices/mat2.txt";
    char* result_fname = "test_matrices/mult_mat1_2.txt";
    Matrix mat1 = read_mat(mat1_fname);  
    Matrix mat2 = read_mat(mat2_fname);
    Matrix true_result = read_mat(result_fname);
    Matrix result = mult(&mat1, &mat2);
    //printf("Result shape: %d %d\n", result.nrows, result.ncols);
    if (equal(&result, &true_result))
        printf("Passed\n");
    else
        printf("Failed\n");
}

void test_frobenius_norm()
{
    // mat 1 has fro norm of 182.2955
    printf("Testing Sq Frobenius Norm: ");
    char* mat1_fname = "test_matrices/mat1.txt";
    Matrix mat1 = read_mat(mat1_fname);
    float result = sq_frobenius_norm(&mat1);
    //printf("Norm is %f\n", result);
    // sqrt is less stable than other operations, need smaller tolerance
    if (fabs(result - (182.2955 * 182.2955)) < 0.01)
        printf("Passed\n");
    else
        printf("Failed\n");
}

// test what we have so far
int main(int argc, char* argv[])
{
    test_add();
    test_transpose();
    test_scalar_mult();
    test_mult();
    test_frobenius_norm();
    test_truncate_cols();
}

