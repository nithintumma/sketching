/*
  An implementation of SVD from Numerical Recipes in C and Mike Erhdmann's lectures
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <stdbool.h>
#include "matrix.h"

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : - fabs(a))

static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1 = (a),maxarg2 = (b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1 = (a),iminarg2 = (b),(iminarg1 < (iminarg2) ? (iminarg1) : iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)


// prints an arbitrary size matrix to the standard output
void printMatrix(double **a, int rows, int cols) {
  int i,j;

  for(i=0;i<rows;i++) {
    for(j=0;j<cols;j++) {
      printf("%.4lf ",a[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

// prints an arbitrary size vector to the standard output
void printVector(double *v, int size) {
  int i;

  for(i=0;i<size;i++) {
    printf("%.4lf ",v[i]);
  }
  printf("\n\n");
}

// calculates sqrt( a^2 + b^2 ) with decent precision
double pythag(double a, double b) {
  double absa,absb;

  absa = fabs(a);
  absb = fabs(b);

  if(absa > absb)
    return(absa * sqrt(1.0 + SQR(absb/absa)));
  else
    return(absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

/*
  Modified from Numerical Recipes in C
  Given a matrix a[nRows][nCols], svdcmp() computes its singular value 
  decomposition, A = U * W * Vt.  A is replaced by U when svdcmp 
  returns.  The diagonal matrix W is output as a vector w[nCols].
  V (not V transpose) is output as the matrix V[nCols][nCols].
*/
int svdcmp(double **a, int nRows, int nCols, double *w, double **v) {
  int flag,i,its,j,jj,k,l,nm;
  double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

  rv1 = malloc(sizeof(double)*nCols);
  if(rv1 == NULL) {
    printf("svdcmp(): Unable to allocate vector\n");
    return(-1);
  }

  g = scale = anorm = 0.0;
  for(i=0;i<nCols;i++) {
    l = i+1;
    rv1[i] = scale*g;
    printf("Iteration %d: %f\n", i, rv1[i]);
    g = s = scale = 0.0;
    if(i < nRows) {
      for(k=i;k<nRows;k++) 
      {
          scale += fabs(a[k][i]);
          //printf("Sub-iter %d %d: %f\n", i, k, scale);
      }
      if(scale) {
          printf("In if Scale: %f\n", scale);
        for(k=i;k<nRows;k++) 
        {
            a[k][i] /= scale;
            //printf("Sub-iter %d %d: %f\n", i, k, a[k][i]);
            s += a[k][i] * a[k][i];
        }
        f = a[i][i];
        g = -SIGN(sqrt(s),f);
        h = f * g - s;
        a[i][i] = f - g;
        //printf("set_ind: %f\n", a[i][i]);
        for(j=l;j<nCols;j++) {
            for(s=0.0,k=i;k<nRows;k++) s += a[k][i] * a[k][j];
            f = s / h;
            for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
        }
        for(k=i;k<nRows;k++) a[k][i] *= scale;
      }
    }
    w[i] = scale * g;
    g = s = scale = 0.0;
    if(i < nRows && i != nCols-1) {
      for(k=l;k<nCols;k++) scale += fabs(a[i][k]);
      printf("Scale: %0.2f\n", scale);
      if(scale)  {
        for(k=l;k<nCols;k++) {
            a[i][k] /= scale;
            s += a[i][k] * a[i][k];
            printf("s: %0.2f\n", s);
        }
        f = a[i][l];
        g = - SIGN(sqrt(s),f);
        h = f * g - s;
        a[i][l] = f - g;
        for(k=l;k<nCols;k++) rv1[k] = a[i][k] / h;
        for(j=l;j<nRows;j++) {
            for(s=0.0,k=l;k<nCols;k++) s += a[j][k] * a[i][k];
            for(k=l;k<nCols;k++) a[j][k] += s * rv1[k];
        }
        for(k=l;k<nCols;k++) a[i][k] *= scale;
      }
    }
    anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));

    printf(".");
    fflush(stdout);
  }
  printf("\nFirst Print after anorm loop\n");
  printMatrix(a, nRows, nCols);

  for(i=nCols-1;i>=0;i--) {
    if(i < nCols-1) {
      if(g) {
    for(j=l;j<nCols;j++)
      v[j][i] = (a[i][j] / a[i][l]) / g;
    for(j=l;j<nCols;j++) {
      for(s=0.0,k=l;k<nCols;k++) s += a[i][k] * v[k][j];
      for(k=l;k<nCols;k++) v[k][j] += s * v[k][i];
    }
      }
      for(j=l;j<nCols;j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
    printf(".");
    fflush(stdout);
  }

  for(i=IMIN(nRows,nCols) - 1;i >= 0;i--) {
    l = i + 1;
    g = w[i];
    for(j=l;j<nCols;j++) a[i][j] = 0.0;
    if(g) {
      g = 1.0 / g;
      for(j=l;j<nCols;j++) {
    for(s=0.0,k=l;k<nRows;k++) s += a[k][i] * a[k][j];
    f = (s / a[i][i]) * g;
    for(k=i;k<nRows;k++) a[k][j] += f * a[k][i];
      }
      for(j=i;j<nRows;j++) a[j][i] *= g;
    }
    else
      for(j=i;j<nRows;j++) a[j][i] = 0.0;
    ++a[i][i];
    printf(".");
    fflush(stdout);
  }

  for(k=nCols-1;k>=0;k--) {
    for(its=0;its<30;its++) {
      flag = 1;
      for(l=k;l>=0;l--) {
    nm = l-1;
    if((fabs(rv1[l]) + anorm) == anorm) {
        printf("Terminating, %f %f\n", anorm, rv1[l]);
      flag =  0;
      break;
    }
    if((fabs(w[nm]) + anorm) == anorm) 
    {
        printf("Terminating 2, %f %f\n", anorm, w[nm]);
        break;
    }
      }
      if(flag) {
    c = 0.0;
    s = 1.0;
    for(i=l;i<=k;i++) {
      f = s * rv1[i];
      rv1[i] = c * rv1[i];
      if((fabs(f) + anorm) == anorm) 
      {
          printf("Terminating 3, %f %f\n", anorm, f);
          break;
      }
      g = w[i];
      h = pythag(f,g);
      w[i] = h;
      h = 1.0 / h;
      c = g * h;
      s = -f * h;
      for(j=0;j<nRows;j++) {
        y = a[j][nm];
        z = a[j][i];
        a[j][nm] = y * c + z * s;
        a[j][i] = z * c - y * s;
      }
    }
      }
      z = w[k];
      if(l == k) {
    if(z < 0.0) {
      w[k] = -z;
      for(j=0;j<nCols;j++) v[j][k] = -v[j][k];
    }
        printf("Terminating l=k, %f\n", anorm);
    break;
      }
      if(its == 29) printf("no convergence in 30 svdcmp iterations\n");
      x = w[l];
      nm = k-1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = pythag(f,1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f))) - h)) / x;
      c = s = 1.0;
      for(j=l;j<=nm;j++) {
    i = j+1;
    g = rv1[i];
    y = w[i];
    h = s * g;
    g = c * g;
    z = pythag(f,h);
    rv1[j] = z;
    c = f/z;
    s = h/z;
    f = x * c + g * s;
    g = g * c - x * s;
    h = y * s;
    y *= c;
    for(jj=0;jj<nCols;jj++) {
      x = v[jj][j];
      z = v[jj][i];
      v[jj][j] = x * c + z * s;
      v[jj][i] = z * c - x * s;
    }
    z = pythag(f,h);
    w[j] = z;
    if(z) {
      z = 1.0 / z;
      c = f * z;
      s = h * z;
    }
    f = c * g + s * y;
    x = c * y - s * g;
    for(jj=0;jj < nRows;jj++) {
      y = a[jj][j];
      z = a[jj][i];
      a[jj][j] = y * c + z * s;
      a[jj][i] = z * c - y * s;
    }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
    printf(".");
    fflush(stdout);
  }
  printf("\n");
  
  free(rv1);
  
  return(0);
}

int main(int argc, char* argv[])
{
    // read in a matrix. convert it, then stuff
    char* fname = "test_matrices/small_svd_mat.txt";
    Matrix A = read_mat(fname);
    print_mat(&A);
    double** a = convert_mat(&A);
    // allocate output arrays
    double* w = malloc(sizeof(double) * A.nrows);
    double** v = malloc(sizeof(double *) * A.ncols);
    for (int i=0; i<A.ncols; i++)
        v[i] = malloc(sizeof(double) * A.ncols);
    if (svdcmp(a, A.nrows, A.ncols, w, v) == 0)
    {
        // print out the singular values
        printf("Singular values\n");
        printVector(w, A.nrows);
        printMatrix(a, A.nrows, A.ncols);
        return 0;
    }
    else
        return 1;
}
