#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"
#include "sketch.h"
#include "svd.h"

struct sketch_arg_struct
{
	Matrix* mat; 
	int row_start;
	int row_end; 
	// sketch size
	int l;
	float alpha; 
	// holds returned sketch
	//TODO: how do we make this a pointer? 
	Matrix sketch; 
};

struct merge_sketch_arg_struct
{
	Matrix* sketch;
	Matrix* mat; 
	// sketch size
	int l;
	float alpha; 
};

// what do we do here? like do we have to copy the matrix a bunch? seems like a waste of time
// maybe we create a new fd_sketch_l function that takes in start and end row, probably faster than making a new matrix each time and copying the data over

// now what should I do? need to create an arg struct for input into 
void* t_sketch_mat(void* arguments)
{
	struct sketch_arg_struct *sketch_args = (struct sketch_arg_struct *) arguments;
	// so what do we need to do here? we just need t
	Matrix sketch_mat = sub_fd_sketch_l(sketch_args->mat, sketch_args->row_start, sketch_args->row_end, sketch_args->l);
	sketch_args->sketch = sketch_mat; 
	return NULL;
}

void* t_merge_sketches(void* arguments)
{
	struct merge_sketch_arg_struct *merge_args = (struct merge_sketch_arg_struct *) arguments;
	update_fd_sketch_l(merge_args->sketch, merge_args->mat, merge_args->l);
	return NULL;
}

//TODO, make same call signature as construct_sketch_l, with additional num_threads
//TODO, make callable from command line run_sketch.c 
int run_t_sketch(char* mat_fname, int l, int num_threads)
{
	// create threads 
	pthread_t thread_ids[num_threads];
	// load the matrix
	Matrix mat = read_mat(mat_fname);
	// create the arg structs 
	struct sketch_arg_struct args[num_threads];
	for (int i=0; i<num_threads; i++)
	{
		args[i].mat = &mat;
		args[i].l = l;
		args[i].alpha = 1.0;
		args[i].row_start = i * mat.nrows / num_threads;
		args[i].row_end = -1 + (i+1) * mat.nrows/num_threads;
		if (i == num_threads - 1)
			args[i].row_end = mat.nrows - 1;
		//printf("%d: sr: %d, er: %d\n", i, args[i].row_start, args[i].row_end);
	}

	for (int i=0; i<num_threads; i++)
	{
		// create an arg struct 
		if(pthread_create(&thread_ids[i], NULL, t_sketch_mat, &args[i])) 
		{
			printf("Failed creating threads\n");
			exit(1);
		}
	}

	// print out the error 
	for (int i=0; i<num_threads; i++)
	{
		if(pthread_join(thread_ids[i], NULL)) 
		{
			printf("Failed joining threads\n");
			exit(1);
		}
	}
	// combine the sketches from each thread 
	int num_sketches = num_threads;
	struct merge_sketch_arg_struct merge_args[num_sketches];
	// points to the possible extra sketch 
	int i; 
	Matrix* extra_sketch; 
	while(num_sketches > 1)
	{
		for (i=0; i<num_sketches/2; i++)
		{
			// construct the args for the thread, then dispatch the thread
			if (num_sketches == num_threads)
			{
				// first time through, use old args
				merge_args[i].l = args[2*i].l; 
				merge_args[i].sketch = &args[2 * i].sketch; 
				merge_args[i].mat = &args[2*i + 1].sketch;
			}
			else
			{
				// need to combine sketches from merge_args
				merge_args[i].l = merge_args[2*i].l; 
				merge_args[i].sketch = merge_args[2*i].sketch; 
				merge_args[i].mat = merge_args[2*i + 1].sketch;
			}
			// dispatch the thread to to do this work 
			if(pthread_create(&thread_ids[i], NULL, t_merge_sketches, &merge_args[i])) 
			{
				printf("Failed creating thread %d\n", i);
				exit(1);
			}
		}
		// join threads 
		for (i=0; i<num_sketches/2; i++)
		{
			if(pthread_join(thread_ids[i], NULL)) 
			{
				printf("Failed joining thread %d\n", i);
				exit(1);
			}
		}

		if (i < ceil(num_sketches/2.0))
		{
			if (num_sketches == num_threads)
				extra_sketch = &args[num_sketches-1].sketch;
			else
				extra_sketch = merge_args[num_sketches-1].sketch;
			update_fd_sketch_l(merge_args[0].sketch, 
								extra_sketch, merge_args[0].l);
		}
		num_sketches /= 2;
	}
	write_mat(merge_args[0].sketch, "test_matrices/t_test_sketch.txt");
	return 0; 
}


// DOESN't WORK brutal what should we do? well we can just do the original sketch 
void test_batched_sketch()
{
    char* fname = "test_matrices/med_svd_mat.txt";
    char* write_fname =  "test_matrices/test_sketch.txt";
    Matrix mat = read_mat(fname);
    int l = 20;
    int batch_size = 20;
    Matrix sketch = batch_fd_sketch_l(&mat, l, batch_size);
    write_mat(&sketch, write_fname); 
}

int main(int argc, char* argv[])
{
	char* fname = "test_matrices/med_svd_mat.txt";
	int l = 30;
	int num_threads = 3;
	if(run_t_sketch(fname, l, num_threads))
	{
		printf("Failed\n");
		return 1;
	}
	return 0;
}
