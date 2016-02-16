// forward declaration of Matrix struct 
struct Matrix;
typedef struct Matrix Matrix; 

double pythag(double a, double b);
// computes SVD decomposition of input matrix mat 
int svd(Matrix* mat, double* w, Matrix* V);
int reorder(Matrix* U, double* w, Matrix* V);
// return largest singular value of matrix
double l2_norm(Matrix* mat);

// sample_svd conversion
double** convert_mat(Matrix* mat);
void print_double_mat(double** mat, int nrows, int ncols);
