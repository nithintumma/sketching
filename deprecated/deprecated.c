// test threading 
struct arg_struct
{
    int* nums;
    int start_ind;
    int end_ind;
    int sum; 
};


void *t_add_nums(void* arguments)
{
    struct arg_struct *args = (struct arg_struct *) arguments; 
    args->sum = 0;
    for (int i=args->start_ind; i<=args->end_ind; i++)
        args->sum += args->nums[i];
    return NULL;
}

int test_threads()
{
    // what do we do here? 
    // construct the int arrays that will be added up, give them start
    int num_threads = 4;
    pthread_t thread_ids[num_threads];
    int num_len = 10;
    int num_arr[num_len];
    for (int i=0; i<num_len;i++)
        num_arr[i] = i+1;
    // calculate offsets, create arg structs, 
    struct arg_struct args[num_threads];
    // calculate offsets 

    int start_inds[num_threads];
    int end_inds[num_threads];
    for (int i=0; i<num_threads; i++)
    {
        start_inds[i] = i * num_len / num_threads;
        if (i == num_threads - 1)
            end_inds[i] = num_len - 1;
        else
            end_inds[i] = -1 + (i+1) * num_len / num_threads;
        // make the struct here 
        args[i].nums = num_arr;
        args[i].start_ind = start_inds[i];
        args[i].end_ind = end_inds[i];
    }

    for (int i=0; i<num_threads; i++)
    {
        // create an arg struct 
        if(pthread_create(&thread_ids[i], NULL, t_add_nums, &args[i])) 
        {
            printf("Failed creating threads\n");
            exit(1);
        }
    }

    // join the threads 
    for (int i=0; i<num_threads; i++)
    {
        if(pthread_join(thread_ids[i], NULL)) 
        {
            printf("Failed joining threads\n");
            exit(1);
        }
    }
    // print out all of the sums, return their sum 
    int sum = 0;
    for (int i=0; i<num_threads; i++)
    {
        printf("sum %d: %d\n", i, args[i].sum);
        sum += args[i].sum;
    }
    printf("Total is %d\n", sum);
    return 0;
}

// returns the index of the first zero row in mat, -1 if none exists
// TODO: return a list of all of the zero valued rows so that we can update the sketch faster 
int zero_row(Matrix* mat)
{
    bool flag; 
    // don't ever break out of this and instead once we reach a zero row, 
    for (int i=0; i<mat->nrows; i++)
    {
        flag = true;
        // return true if B has a zero valued row  
        for (int j=0; j<mat->ncols; j++)
        {
            // if every one is zero we want to break and return true
            if (!(FLOAT_EQ(get_ind(mat, i, j), 0.0)))
            {
                flag = false;
                break;
            }
        }
        if (flag)
            return i;
    }
    return -1;
}

Matrix fd_sketch(Matrix* mat, int l)
{
    // check that l is less than m
    if (l >= mat->nrows)
    {
        printf("Cannot sketch into more rows than input\n");
        exit(1);
    } 
    // would be nice if l was a multiple of 2 as well
    if (((l % 2) != 0) && l > 2)
    {
        printf("Require sketch size to be a multiple of 2, >= 4\n");
        exit(1);
    }
    // decide which svd implementation we are going to use
    

    int j, zero_ind, num_zero_rows, zrow_ind;
    int i = 0;
    double del; 
    Matrix sketch, V, V_trunc, V_trunc_t, E;  
    sketch = zeros(l, mat->ncols);

    // svals will be the singular values of the sketch matrix
    // TODO: make this min(l, ncols) (and enforce l <= nrows)
    int svals_len = (int)fmin(mat->ncols, l);
    // TODO: need to allocate for ncols even though we only use the first l 
    double* svals = malloc(sizeof(double) * mat->ncols);
    // TODO: make V ncols X min(ncols, l)
    // TODO: should this be here or only in svd 
    V = zeros(mat->ncols, mat->ncols);

    // run while we have rows left to sketch 

    while(i < mat->nrows)
    {
        //TODO: do we need to zero out svals, U, V each time? dont think so 
        zero_ind = zero_row(&sketch);
        if (zero_ind >=0)
        {
            // insert the current row of mat into a zero valued row of B
            //TODO: optimize this with a memcpy?
            for (j=0; j<mat->ncols; j++)
                set_ind(&sketch, zero_ind, j, get_ind(mat, i, j));
            i++;
        }
        else
        {
            // sketch has no zero valued rows, run sketching procedure
            // svd overwrites sketch with the U matrix
            // TODO: does this matter            
            if (svd(&sketch, svals, &V, true) == 0)
            {
                V_trunc = truncate_cols_2(&V, svals_len);
                // should pick out the middle singular value
                del = svals[(int)floor(svals_len/2)];
                // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
                for(j=0; j<svals_len; j++)
                {
                    svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
                }

                E = diag(svals, svals_len);                 
                //TODO: V is too large, we need to figure out which columns to keep?
                // reduced SVD: keep the first l columns, but how?
                //we could just change the ncols of V, while keeping the data there
                V_trunc_t = transpose(&V_trunc); 
                // new sketch is EV^T
                // TODO: can optimize since E is diagonal
                sketch = mult(&E, &V_trunc_t);
            }
            else
            {
                printf("SVD Failed\n");
                exit(1);
            }
        }
    }
    // REPEAT ABOVE LOGIC to sketch
    // so why is our program so brutal? 
    // shoud we only do this if we don't have any zero rows at the end?
    zero_ind = zero_row(&sketch);
    if (zero_ind == -1)
    {
        if (svd(&sketch, svals, &V, false) == 0)
        {
            // do we need to truncate the columns of V? Yes for now 
            V_trunc = truncate_cols_2(&V, svals_len);
            // should pick out the middle singular value
            del = svals[(int)floor(l/2)];
            // E = diag(sqrt(max(svals ** 2 - delta ** 2, 0)))
            for(j=0; j<svals_len; j++)
            {
                svals[j] = sqrt(fmax(SQUARE(svals[j]) - SQUARE(del), 0.0));
            }

            E = diag(svals, svals_len);                 
            //TODO: V is too large, we need to figure out which columns to keep?
            // reduced SVD: keep the first l columns, but how?
            //we could just change the ncols of V, while keeping the data there
            V_trunc_t = transpose(&V_trunc); 
            // new sketch is EV^T
            // TODO: can optimize since E is diagonal
            sketch = mult(&E, &V_trunc_t);

        }
        else
        {
            printf("SVD Failed\n");
            exit(1);
        }
    }
    // TODO: clean up everything 
    free(svals);
    free_mat(&V);
    free_mat(&V_trunc);
    free_mat(&E);
    free_mat(&V_trunc_t);
    return sketch;
}

// make sure that we satisfy the bound on reconstruction of covariance matrix
int test_sketch(char* fname, int l, bool write, char* write_fname)
{
    Matrix mat = read_mat(fname);
    // run time code here 
    Matrix sketch = fd_sketch(&mat, l);
    //
    if (write)
        write_mat(&sketch, write_fname); 

    double err = recon_error(&mat, &sketch);
    printf("Sq frobenius norm: %f\n", sq_frobenius_norm(&mat));
    double bound = (2.0 * sq_frobenius_norm(&mat)) / (float) l;
    printf("Err: %f, Bound: %f\n", err, bound);
    printf("Shape %d %d, L: %d\n", sketch.nrows, sketch.ncols, l);
    free_mat(&mat);
    free_mat(&sketch);
    if (bound >= err)
        return 1;
    else
        return 0;
}
