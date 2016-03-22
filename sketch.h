// forward declaration of Matrix struct 
struct Matrix;
typedef struct Matrix Matrix; 

// returns the index of the first zero row in mat, -1 if none exists
int zero_row(Matrix* mat);


Matrix batch_fd_sketch_l(Matrix* mat, int l, int batch_size);

// returns a FD sketch of input matrix with l rows
Matrix fd_sketch_l(Matrix* mat, int l);
// runs fd_sketch_l on submatrix of mat 
Matrix sub_fd_sketch_l(Matrix* mat, int start_row, int end_row, int l);
// update sketch in place with rowws of mat
void update_fd_sketch_l(Matrix* sketch, Matrix* mat, int l);

// driver function to read mat, construct sketch, and write mat 
int construct_sketch_l(char* fname, int l, float alpha, bool write, char* write_fname, bool check);

// calculate covariance reconstruction error
double recon_error(Matrix* mat, Matrix* sketch);
