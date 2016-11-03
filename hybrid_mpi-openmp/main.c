/* Comp 4510 - Hybrid MPI-OpenMP Array Addition Example
 *
 * This code sums up the elements of a vector of floats, using a 2-stage reduction.
 * First, the array is partitioned and distributed among the MPI processes.
 * Next, each process uses OpenMP to compute the partial sum of its chunk.
 * Finally, all processes collaborate to combine the partial sums using an MPI reduction operation.
 *
 * The number of MPI processes and OpenMP threads are set in the myjob file.
 * 
 * Note: for this to work correctly, N (below) must be divisible by
 * the number of processes being used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

/* Constants */

#define N (1 << 20)   // (2^20) size of vector (must be divisible by number of processes)

/* Prototypes */

void rand_array(float *array, int len);
float sum_array(float *array, int len);

/* Functions */

// Fills a vector with random floats in the range [0, 1)
void rand_array(float *array, int len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        //keep the floats small so we can scale up N without worrying about overlow
        array[i] = (float) rand() / RAND_MAX;
    }    
}

// Returns the sum of the elements in the given array.
float sum_array(float *array, int len)
{
    float total = 0;

    //use OpenMP to reduce the current proc's chunk
#pragma omp parallel for reduction(+:total)
    for (int i = 0; i < len; i++)
    {
        total += array[i];
    }

    return total;
}

int main(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_time; // use these for timing
    double stop_time;
    
    float *array; // Note: we'll initialize this only on proc 0 (in "if" block below) avoid wasting memory on the other hosts
    int chunk_size = N / num_procs;
    float chunk[chunk_size]; // everybody needs this one...
    float final_sum;
    float partial_sum;

    // Initialize the array
    if (!my_rank)
    {
        array = (float *) malloc(sizeof(float) * N);
        srand(time(NULL));
        rand_array(array, N);
        printf("Summing %d floats...\n", N);
        start_time = MPI_Wtime();
    }

    // Scatter the data among the MPI processes
    MPI_Scatter(array, chunk_size, MPI_FLOAT, chunk, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Perform a 2-stage reduction. */
    // 1. Each proc computes a partial sum by adding up its chunk of the array using OpenMP.
    partial_sum = sum_array(chunk, chunk_size);
    // 2. All procs cooperate in an MPI reduction operation to combine their partial results.
    MPI_Reduce(&partial_sum, &final_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Output the results and cleanup
    if (!my_rank)
    {
        stop_time = MPI_Wtime();
        printf("Complete!\n");
        printf("Final sum: %f\n", final_sum);
        printf("Total time: %f sec\n", stop_time - start_time);

        free(array);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
