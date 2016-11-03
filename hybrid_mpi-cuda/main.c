#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include "crunch.h"

#define PRINT_VECS 0
#define MAX_RAND 10

void gen_data(int *buf, int len)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < len; i++)
    {
        buf[i] = rand() % MAX_RAND;
    }    
}

void print_vec(const char *label, int *vec, int len)
{
#if PRINT_VECS
    printf("%s", label);
    
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%d ", vec[i]);
    }
    printf("\n\n");
#endif
}

int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: hybrid <y, where n = 2^y>\n");
        exit(1);
    }
    
    return 1 << atoi(argv[1]);
}

int sum_array(int *data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += data[i];
    }

    return sum;
}

void print_hostname()
{
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(proc_name, &len);
    printf("Using system: %s\n", proc_name);
}

int main(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = parse_args(argc, argv);
    print_hostname();
    
    int *data;
    int seq_sum;
    if (!my_rank)
    {
        data = (int *) malloc(n * sizeof(int));
        gen_data(data, n);
        
        print_vec("Initial Vector: ", data, n);
        seq_sum = sum_array(data, n);
    }
    
    int chunk_size = pow(2, floor(log2f(n / num_procs)));
    int extras = n - chunk_size * num_procs;
    if (!my_rank)
    {
        data[0] += sum_array(data + chunk_size * num_procs, extras);
    }
    n -= extras;
    
    int chunk[chunk_size];
    MPI_Scatter(
        data,
        chunk_size,
        MPI_INT,
        chunk,
        chunk_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
        );
    
    int partial = crunch(my_rank, chunk, chunk_size);
    int sum;
    MPI_Reduce(
        &partial,
        &sum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
        );

    if (!my_rank)
    {
        printf("Computed sum: %d\n", sum);
        printf("Correct sum: %d\n", seq_sum);
        free(data);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
