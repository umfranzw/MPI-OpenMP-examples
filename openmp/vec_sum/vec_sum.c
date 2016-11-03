/* Comp 4510 - Vector Sum Example (OpenMP)
 * This program sums up the elements in a randomly
 * generated vector of integers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N (1 << 4) // (2^4) size of vector (does *not* need to be divisible by number
                   // of threads - openmp takes care of this for us as part of the
                   // reduction operation - see main())
#define PRINT_VECS 1  // flag so we can turn off printing when N is large
#define MAX_RAND 100  // max value of elements generated for array

/* Prototypes */
void init_vec(int *vec, int len);
void print_vec(const char *label, int *vec, int len);
int sum_chunk(int *vec, int low_index, int high_index);

/* Functions */

// Fills a vector with random integers in the range [0, MAX_RAND)
void init_vec(int *vec, int len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        vec[i] = rand() % MAX_RAND;
    }    
}

// Prints the given vector to stdout
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

int main(int argc, char *argv[])
{
    // Declare arrays for data and results
    // Note: all threads share one copy of this
    int vec[N]; // input vector
    double start_time; // use these for timing
    double stop_time;

    // Fill the vector and start the timer
    printf("N: %d\n", N);
    srand(time(NULL));
    init_vec(vec, N);
    print_vec("Initial vector:\n", vec, N);
    start_time = omp_get_wtime(); // can use this function to grab a
                                  // timestamp (in seconds)

    // Fork into separate threads to compute the sum of the elements.
    // Note: OpenMP automatically sums up the partial results from each thread
    // and deposits the final result in the sum variable when the loop finishes.
    int sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++)
    {
        sum += vec[i];
    }
    // Here all threads join back together

    // Print result & timing info
    stop_time = omp_get_wtime();
    printf("Final result: %d\n", sum);
    printf("Total time (sec): %f\n", stop_time - start_time);

    return EXIT_SUCCESS;;
}
