// currently store row-wise, but can change 
typedef struct Matrix
{
    unsigned int ncols;
    unsigned int nrows;
    double* matrix;
} Matrix;

// construc/modify matrix 
Matrix init_mat(int rows, int cols);
void free_mat(Matrix* mat);
Matrix zeros(int rows, int cols);
Matrix diag(double* diagonal, int rows);
Matrix rand_matrix(int rows, int cols);
Matrix eye(int rows);
void truncate_cols(Matrix** mat, int ncols);
Matrix truncate_cols_2(Matrix* mat, int ncols);

double get_ind(Matrix* mat, int x, int y);
void set_ind(Matrix* mat, int x, int y, double val); 

// matrix IO 
void print_mat(Matrix* mat);
void write_mat(Matrix* mat, char* fname);
Matrix read_mat(char* fname);

// Math functions 
Matrix add(Matrix* mat1, Matrix* mat2);
Matrix subtract(Matrix* mat1, Matrix* mat2);
Matrix mult_scalar(Matrix* mat, double c);
Matrix transpose(Matrix* mat);
Matrix mult(Matrix* mat1, Matrix* mat2);
double sq_frobenius_norm(Matrix* mat);
bool equal(Matrix* mat1, Matrix* mat2);

// interface with sample_svd matrix format
double** convert_mat(Matrix* mat);
void print_s_mat(double** mat, int nrows, int ncols);
