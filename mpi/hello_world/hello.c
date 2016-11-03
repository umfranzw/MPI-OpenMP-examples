/* Comp 4510 - MPI Hello World Example
 * Standard hello, world program - the myjob script controls
 * how many processes are launched.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    // Declare process-related vars (note: each process has its own copy)
    // and initialize MPI
    int my_rank;
    int num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //grab this process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //grab the total num of processes

    // Note that the output may appear in a different order each time
    // the program is run. The order depends on each processor's current
    // load, and how busy the interconnection network is.
    printf("Hello from process %d! (%d processes running)\n", my_rank, num_procs);

    // Shutdown MPI (important - don't forget!)
    MPI_Finalize();

    return EXIT_SUCCESS;
}
