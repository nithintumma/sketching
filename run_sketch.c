#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#include "sketch.h"

// TODO: should we give the option of OG svd?? dont want to
// takes command line arguments and runs construct_sketch_l from sketch.c

int main(int argc, char* argv[])
{
    clock_t begin, end;
    double time_spent;  
    //gettimeofday(&begin, NULL);
    begin = clock();

    extern char *optarg;
    extern int optind;
    int c, err = 0;
    static char usage[] = "usage: -f fname [-w write_fname] -l l\n";
    char *fname, *write_fname;
    bool fflag, wflag, lflag, aflag = 0;
    float a;  
    int l;
    while((c = getopt(argc, argv, "f:w:l:a")) != -1)
    {
        switch (c) 
        {
            case 'f':
                    fflag = 1;
                    fname = optarg;
                    break;
            case 'w':
                    wflag = 1;
                    write_fname = optarg;
                    break;
            case 'l':
                    lflag = 1;
                    l = atoi(optarg);
                    break;
            case 'a':
                    aflag = 1;
                    a = atof(optarg);
                    break;
        }
    }
    //printf("Command line args: %s %s %d\n", fname, write_fname, l);
    if (!(fflag && lflag))
    {
        printf("Parsing failed\n%s", usage);
        exit(1);
    }
    // read in a matrix, compute its sketch, then test accuracy?
    bool check = false; 
    if (!aflag)
        a = 1.0;
    if (construct_sketch_l(fname, l, a, wflag, write_fname, check) == 0)
    {
        end = clock();
        time_spent = (double) (end - begin)/ CLOCKS_PER_SEC;
        printf("%f ", time_spent);
        return 0;
    }
    else
        return 1;    
}
