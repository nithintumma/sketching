// forward declaration of Matrix struct 
struct Matrix;
typedef struct Matrix Matrix; 

// returns the index of the first zero row in mat, -1 if none exists
int zero_row(Matrix* mat);

// returns a FD sketch of input matrix with l rows
Matrix fd_sketch(Matrix* mat, int l);
