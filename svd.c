/* SVD implementation adapted from Numerical Recipes: 
 * http://numerical.recipes/webnotes/nr3web2.pdf
 */

//TODO: compare this implementation to svd.c in outer dir
//TODO: why are we returning nans and infs? 
//TODO: change doubles to doubles 
//TODO: comapre this implementation to LAPACK Fortran implementation 

#include <time.h>
#include <math.h> 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix.h"

#define FLOAT_EQ(a, b) (fabs(a - b) < 0.000001)
//#define FLOAT_EQ(a, b) (a == b)
#define SIGN(a, b) ((b) > 0.0 ? fabs(a): -fabs(a))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))
#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define SQR(a)     a * a

double pythag(double a, double b)
{
    double aba = fabs(a);
    double abb = fabs(b);
    if (aba > abb)
        return (aba * sqrt(1.0 + SQR(abb/aba)));
    else
        return (abb == 0.0 ? 0.0: abb * sqrt(1.0 + SQR(aba/abb)));
}

/*
 returns the SVD of matrix mat with U stored in mat
 singular values in the array w
 and 
*/
int svd(Matrix* mat, double* w, Matrix* V)
{
    double anorm, c, f, g, h, s, scale, x, y, z, val;
    scale = 0.0;
    anorm = 0.0;
    g = 0.0;
    double *rv1; 
    int n, m, nm, its; 
    bool flag; 
    n = mat->ncols;
    m = mat->nrows;
    rv1 = (double*) malloc(sizeof(double) * n);
    
    //TODO: add checks for NULL pointers everywhere 
    int i, j, k, l, jj;
    for (i=0;i<n;i++)
    {
        //TODO: other says + 2
        l = i+1;
        rv1[i] = scale*g; 
        g = s = scale = 0.0;
        if (i < m)
        {
            for (k = i; k<m; k++)
            {
                //printf("+scale: %0.2f\n", fabs(get_ind(mat, k, i)));
                scale += fabs(get_ind(mat, k, i));
                //printf("Sub-iter %d: %f\n", k, scale);
            }
            if (scale)
            {
                for (k=i;k<m;k++)
                {
                    set_ind(mat, k, i, get_ind(mat, k, i)/scale);
                    //printf("Sub-iter %d %d: %f\n", i, k, get_ind(mat, k, i));
                    s += get_ind(mat, k, i) * get_ind(mat, k, i);
                }
                f = get_ind(mat, i, i);
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                set_ind(mat, i, i, f-g);
                //printf("set_ind: %f\n", get_ind(mat, i, i));
                //TODO: change this to l?
                for(j=l; j<n; j++)
                {
                    for(s=0.0,k=i;k<m;k++)
                        s += get_ind(mat, k, i) * get_ind(mat, k, j);
                    f = s/h;
                    for (k=i; k<m; k++)
                    {
                        val = f * get_ind(mat, k, i);
                        set_ind(mat, k, j, get_ind(mat, k, j)+val);
                    }
                }
                for (k=i; k<m; k++)
                    set_ind(mat, k, i, get_ind(mat, k, i)*scale);

            }
        }
        w[i] = scale * g;
        //TODO: why is scale so messed up on the flip? 
        //printf("w[i]: %0.2f, %0.2f\n", scale, g);
        g = s = scale = 0.0;
        if((i< m) && (i+1 != n))
        {
            //TODO: change this to l?
            for (k=l; k<n; k++)
                scale += fabs(get_ind(mat, i, k));
            //TODO: change this to float eq comparison?
            if (scale)
            {
                //TODO: change this to l?
                for (k=l; k<n; k++)
                {
                    set_ind(mat, i, k, get_ind(mat, i, k)/scale);
                    s += get_ind(mat, i, k) * get_ind(mat, i, k);
                }
                f = get_ind(mat, i, l);
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                set_ind(mat, i, l, f-g);
                for(k=l; k<n; k++)
                    rv1[k] = get_ind(mat, i, k)/h;

                for(j=l;j<m;j++)
                {
                    for(s=0.0,k=l;k<n;k++)
                        s += get_ind(mat, j, k) * get_ind(mat, i, k);
                    for(k=l; k<n; k++)
                    {
                        val = s * rv1[k];
                        set_ind(mat, j, k, get_ind(mat, j, k) + val);
                    }
                }
                for(k=l; k<n; k++)
                    set_ind(mat, i, k, get_ind(mat, i, k) * scale);

            }
        }
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }

    // original says 'accumulation of right hand transformations'
    for(i=n-1; i>=0; i--)
    {
        if(i < n-1)
        {
            if(g)
            {
                for (j=l; j<n; j++)
                {
                    val = (get_ind(mat, i, j)/get_ind(mat, i, l))/g;
                    set_ind(V, j, i, val);
                }
                for (j=l; j<n; j++)
                {
                    for (s=0.0, k=l; k<n;k++)
                        s+= get_ind(mat, i, k) * get_ind(V, k, j);
                    for (k=l; k<n;k++)
                    {
                        val = s * get_ind(V, k, i);
                        set_ind(V, k, j, get_ind(V, k, j) + val); 
                    }
                }
            }
            for (j=l; j<n; j++)
            {
                set_ind(V, i, j, 0.0);
                set_ind(V, j, i, 0.0);
            }
        }
        set_ind(V, i, i, 1.0);
        g = rv1[i];
        l = i;
    }
    // SEEMS GOOD TO HERE

    // original says 'accumulation of left hand transformations'
    for (i=MIN(m, n)-1; i>=0; i--)
    {
        l = i+1;
        g = w[i];
        for (j=l; j<n; j++)
            set_ind(mat, i, j, 0.0);
        if (g)
        {
            g = 1.0/g;
            for (j=l; j<n; j++)
            {
                for (s=0.0, k=l; k<m;k++)
                    s += get_ind(mat, k, i) * get_ind(mat, k, j); 
                f = (s/get_ind(mat, i, i)) * g;
                for (k=i; k<m; k++)
                {
                    val = f * get_ind(mat, k, i);
                    set_ind(mat, k, j, get_ind(mat, k, j) + val);
                }
            }
            for (j=i; j<m;j++)
                set_ind(mat, j, i, get_ind(mat, j, i) * g);
        }
        else
        {
            for(j=i; j<m; j++)
                set_ind(mat, j, i, 0.0);
        }
        set_ind(mat, i, i, get_ind(mat, i, i) + 1);
    }
    // original says 'diag. of the bidiagonal form'
    for (k=n-1; k>=0; k--)
    {
        for (its=0; its<30; its++)
        {
            flag = 1;
            for (l=k; l>=0; l--)
            {
                nm = l-1;
                //sample has the check on rv1[l] here
                // we had on w[nm]
                //printf("L: %d, RV: %0.2f, W: %0.2f\n", l, rv1[l], w[nm]);
                if (FLOAT_EQ((fabs(rv1[l]) + anorm), anorm))
                //if(fabs(rv1[l] + anorm) == anorm)
                //if (l==0 || (FLOAT_EQ(fabs(rv1[l]), 0.0)))
               {
                    //printf("Terminating, %f %f\n", anorm, rv1[l]);
                    //printf("Flag OFF\n");
                    flag = 0;
                    break;
                }
                if (FLOAT_EQ((fabs(w[nm]) + anorm), anorm))
                //if((fabs(w[nm]) + anorm) == anorm)
                //if (FLOAT_EQ(fabs(w[nm]), 0.0))
                {
                    //printf("Terminating 2\n");
                    break;
                }
            }
            if (flag)
            {
                c = 0.0;
                s = 1.0;
                for (i=l; i<k+1; i++)
                {
                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    //TODO: everyone else has fabs + anorm == anorm
                    if (FLOAT_EQ((fabs(f) + anorm), anorm))
                    //if ((fabs(f) + anorm) == anorm)
                    //if (FLOAT_EQ(fabs(f), 0.0))
                    {
                        //printf("Terminating 3\n");
                        break;
                    }
                    g = w[i];
                    h = pythag(f, g);
                    w[i] = h;
                    h = 1.0/h;
                    c = g * h;
                    s = -f* h;
                    for (j=0; j<m; j++)
                    {
                        y = get_ind(mat, j, nm);
                        z = get_ind(mat, j, i);
                        set_ind(mat, j, nm, y*c + z*s);
                        set_ind(mat, j, i, z*c - y*s);
                    }
                }
            }
            z = w[k];
            //printf("Z: %0.2f\n", z);
            if (l == k)
            {
                //printf("L is equal to k\n");
                if ( z < 0.0)
                {
                    w[k] = -z;
                    for (j=0; j<n; j++)
                        set_ind(V, j, k, -get_ind(V, j, k));
                }
                break;
            }
            if (its == 29)
            {
                printf("Hasn't converged after 30 iterations\n");
                //printf("%0.2f\n", fabs(f));
            }
            // TODO: figure out the singular values 
            x = w[l];
            //for (int ff=0; ff<m; ff++)
            //    printf(" %0.2f ", w[ff]);
            //printf("\nL: %d\n", l);
            nm = k-1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y-z) * (y+z) + (g-h) * (g+h))/(2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x-z) * (x+z) + h * ((y/(f + SIGN(g, f))) - h)) / x;
            c = s = 1.0;
            //printf("starting vals: %0.2f, %0.2f, %0.2f, %0.2f", f, g, h, y);
            //printf("X: %0.2f, Y: %0.2f, Z: %0.2f\n", x, y, z);
            //printf("F: %0.2f\n", f);
            for (j=l; j <=nm; j++)
            {

                i = j+1;
                g = rv1[i];
                y = w[i];
                h = s*g;
                g = c*g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f/z;
                s = h/z;
                f = x*c + g*s;
                g = g*c - x*s;
                h = y*s;
                y *= c;
                //printf("Some vals: %0.2f %0.2f %0.2f %0.2f\n", z, s, c, y);
                for (jj = 0; jj<n; jj++)
                {
                    x = get_ind(V, jj, j);
                    z = get_ind(V, jj, i);
                    set_ind(V, jj, j, x*c + z*s);
                    set_ind(V, jj, i, z*c - x*s);
                }
                z = pythag(f, h);
                w[j] = z;
                if (z)
                {
                    z = 1.0/z;
                    c = f*z;
                    s = h*z;
                }
                f = c*g + s*y;
                x = c*y - s*g;


                for (jj=0; jj<m; jj++)
                {
                    y = get_ind(mat, jj, j);
                    z = get_ind(mat, jj, i);
                    set_ind(mat, jj, j, y*c + z*s);
                    set_ind(mat, jj, i, z*c - y*s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    // TODO: cleanup (free arrays etc.)
    return 0;
}

/*
Reorder the columns of U and V be decreasing magnitude of singular values in w
*/
int reorder(Matrix* U, double* w, Matrix* V)
{
    int i, j, k, s, inc;
    inc = 1;
    double sw;
    int n = U->ncols;
    int m = U->nrows;
    double* su = malloc(sizeof(double) * m);
    double* sv = malloc(sizeof(double) * n);
    do 
    {
        inc *= 3;
        inc++;
    } while(inc <= n);
    
    do
    {
        inc /= 3;
        for(i = inc; i<n; i++)
        {
            sw = w[i];
            for (k=0; k<m; k++)
                su[k] = get_ind(U, k, i);
            for (k=0; k<n; k++)
                sv[k] = get_ind(V, k, i);
            j = i;
            while(w[j - inc] < sw)
            {
                w[j] = w[j - inc];
                for(k = 0; k<m; k++)
                    set_ind(U, k, j, get_ind(U, k, j-inc));
                for (k = 0; k<n; k++)
                    set_ind(V, k, j, get_ind(V, k, j-inc));
                j -= inc;
                if (j < inc)
                    break;
            }
            w[j] = sw;
            for(k=0; k<m; k++)
                set_ind(U, k, j, su[k]);
            for(k=0; k<n; k++)
                set_ind(V, k, j, sv[k]);
        }

    } while(inc > 1);
    //TODO: flip signs? do we need this? 
    return 0;
}



// something's wrong

int test(char* fname)
{
    Matrix mat = read_mat(fname);
    double* w = (double*) malloc(sizeof(double) * mat.ncols);
    printf("Input Matrix\n");
    print_mat(&mat);
    Matrix V = zeros(mat.ncols, mat.ncols);
    if (svd(&mat, w, &V) == 0)
    {
        if (reorder(&mat, w, &V) == 0)
        {
            printf("Singular values\n");
            for (int i=0; i<mat.ncols; i++)
                printf("%0.4f ", w[i]);
            printf("\nLeft Singular Vectors\n");
            print_mat(&mat);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char* argv[])
{
    // read in random matrix 
    // compute svd
    char* fname = "test_matrices/large_svd_mat.txt";
    if (test(fname) == 0)
    {
        printf("Success\n");
        return 0;
    }
    // test did not return 0
    printf("Failed\n");
    return 1;
}
