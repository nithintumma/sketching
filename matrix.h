typedef struct 
{
    unsigned int ncols;
    unsigned int nrows;
    // does this need to be a pointer to a pointer? 
    float* matrix;
} Matrix;

// construct matrix 
Matrix init_mat(int rows, int cols);
Matrix zeros(int rows, int cols);
Matrix rand_mat(int rows, int cols);
Matrix eye(int rows);

float get_ind(Matrix* mat, int x, int y);
void set_ind(Matrix* mat, int x, int y, float val); 

// matrix IO 
void print_mat(Matrix* mat);
void write_mat(Matrix* mat, char* fname);

// Math functions 
Matrix add(Matrix* mat1, Matrix* mat2);
Matrix mult_scalar(Matrix* mat, float c);
Matrix transpose(Matrix* mat);
Matrix mult(Matrix* mat1, Matrix* mat2);
float frobenius_norm(Matrix* mat);
bool equal(Matrix* mat1, Matrix* mat2);
