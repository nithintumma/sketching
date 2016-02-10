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

#define FLOAT_EQ(a, b) (fabs(a - b) < 0.0001)
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

// HOW DO I DEBUG THIS CODE? 
//return an array of matrices representing the three matrices of the SVD
double* svd(Matrix* mat)
{
    // n = mat->ncols, m = mat->nrows
    // 
    double anorm, c, f, g, h, s, scale, x, y, z, val;
    scale = 0.0;
    anorm = 0.0;
    g = 0.0;
    double *rv1, *w; 
    int n, m, nm, its; 
    bool flag; 
    n = mat->ncols;
    m = mat->nrows;
    rv1 = (double*) malloc(sizeof(double) * n);
    // vector that will hold singular values
    w = (double*) malloc(sizeof(double) * n);
    // square matrix holding right vectors (ncols**2)
    Matrix V = zeros(n, n);
    
    //TODO: add checks for NULL pointers everywhere 
    int i, j, k, l, jj;
    for (i=0;i<n;i++)
    {
        //TODO: other says + 2
        l = i+1;
        rv1[i] = scale*g; 
        printf("Iteration %d: %f\n", i, rv1[i]);
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
                printf("In if Scale: %f\n", scale);
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
            printf("Scale: %0.2f\n", scale);
            //TODO: change this to float eq comparison?
            if (scale)
            {
                //TODO: change this to l?
                for (k=l; k<n; k++)
                {
                    set_ind(mat, i, k, get_ind(mat, i, k)/scale);
                    //printf("Set_ind: %0.2f\n", get_ind(mat, i, k));
                    s += get_ind(mat, i, k) * get_ind(mat, i, k);
                    printf("s: %0.2f\n", s);
                }
                f = get_ind(mat, i, l-1);
                g = -SIGN(sqrt(s), f);
                h = f*g - s;
                set_ind(mat, i, l-1, f-g);
                for(k=l-1; k<n; k++)
                    rv1[k] = get_ind(mat, i, k)/h;
                for(j=l-1;j<m;j++)
                {
                    for(s=0.0,k=l-1;k<n;k++)
                        s += get_ind(mat, j, k) * get_ind(mat, i, k);
                    for(k=l-1; k<n; k++)
                    {
                        val = s * rv1[k];
                        set_ind(mat, j, k, get_ind(mat, j, k) + val);
                    }
                    for(k=l-1; k<n; k++)
                        set_ind(mat, i, k, get_ind(mat, i, k) * scale);

                }
            }
        }
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }
    printf("First print after the anorm loop\n");
    print_mat(mat);
    // original says 'accumulation of right hand transformations'
    for(i=n-1 ;i>=0; i--)
    {
        if(i < n-1)
        {
            if(g != 0.0)
            {
                for (j=l; j<n; j++)
                {
                    val = (get_ind(mat, i, j)/get_ind(mat, i, l))/g;
                    set_ind(&V, j, i, val);
                }
                for (j=l; j<n; j++)
                {
                    for (s=0.0, k=l; k<n;k++)
                        s+= get_ind(mat, i, k) * get_ind(&V, k, j);
                    for (k=l; k<n;k++)
                    {
                        val = s * get_ind(&V, k, i);
                        set_ind(&V, k, j, get_ind(&V, k, j) + val); 
                    }
                }
            }
            for (j=l; j<n; j++)
            {
                set_ind(&V, i, j, 0.0);
                set_ind(&V, j, i, 0.0);
            }
        }
        set_ind(&V, i, i, 1.0);
        g = rv1[i];
        l = i;
    }

    // original says 'accumulation of left hand transformations'
    for (i=MIN(m, n)-1; i>=0; i--)
    {
        l = i+1;
        g = w[i];
        for (j=l; j<n; j++)
            set_ind(mat, i, j, 0.0);
        if (g != 0.0)
        {
            g = 1.0/g;
            for (j=l; j<n; j++)
            {
                for (s=0.0, k=l; k<n;k++)
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
                if (l==0 || (FLOAT_EQ(fabs(rv1[l]), 0.0)))
               {
                    flag = 0;
                    break;
                }
                if (FLOAT_EQ(fabs(w[nm]), 0.0))
                    break;
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
                    if (FLOAT_EQ(fabs(f), 0.0))
                        break;
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
            if (l == k)
            {
                if ( z < 0.0)
                {
                    w[k] = -z;
                    for (j=0; j<n; j++)
                        set_ind(&V, j, k, -get_ind(&V, j, k));
                }
                break;
            }
            if (its == 29)
            {
                printf("Hasn't converged after 30 iterations\n");
                printf("%0.2f\n", fabs(f));
                exit(1);
            }
            x = w[l];
            nm = k-1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y-z) * (y-z) + (g-h) * (g-h))/(2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x-z) * (x-z) + h * ((y/(f + SIGN(g, f))) - h)) / x;
            c = s = 1.0;
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
                y = y*c;
                for (jj = 0; jj<n; jj++)
                {
                    x = get_ind(&V, jj, j);
                    z = get_ind(&V, jj, i);
                    set_ind(&V, jj, j, x*c + z*s);
                    set_ind(&V, jj, i, z*c - x*s);
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
                    set_ind(mat, jj, j, y*c - z*s);
                    set_ind(mat, jj, i, z*c - y*s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    // TODO: cleanup (free arrays etc.)
    return w;
}


//TODO: test other implementation of SVD 
int main(int argc, char* argv[])
{
    // read in random matrix 
    // compute svd
    char* fname = "test_matrices/small_svd_mat.txt";
    Matrix mat = read_mat(fname);
    print_mat(&mat);
    double* s_vals = svd(&mat);
    for (int i = 0; i<mat.ncols; i++)
        printf("%f\n", s_vals[i]);
    return 0;
}
