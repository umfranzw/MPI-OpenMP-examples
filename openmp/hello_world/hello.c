/* Comp 4510 - Hello World Example
 * Standard Hello, world example in OpenMP. The number of
 * threads can be controlled by modifying the myjob file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

int main(int argc, char *argv[])
{
    // -- only one thread is running here --
    printf("Hello at the start from thread id %d (%d thread running).\n\n",
           omp_get_thread_num(), omp_get_num_threads());
    
#pragma omp parallel
    {
        // -- multiple threads are running here --
        printf("Hello inside the pragma from thread id %d (%d threads running).\n",
           omp_get_thread_num(), omp_get_num_threads());
    }
    // -- one thread from this point on --
    printf("\nHello at the end from thread id %d (%d thread running).\n\n",
           omp_get_thread_num(), omp_get_num_threads());

    return EXIT_SUCCESS;
}
