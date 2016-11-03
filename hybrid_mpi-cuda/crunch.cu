#include "crunch.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately (makes debugging easier)
    }
}

int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}

int crunch(int my_rank, int *chunk, int chunk_size)
{
    const int block_threads = get_max_block_threads();

    int *dev_chunk;
    cudaMalloc(&dev_chunk, chunk_size * sizeof(int));
    cudaMemcpy(dev_chunk, chunk, chunk_size * sizeof(int), cudaMemcpyHostToDevice);
    int *dev_result;
    cudaMalloc(&dev_result, sizeof(int));

    int blk_threads;
    int num_blks;
    if (chunk_size / 2 <= block_threads)
    {
        blk_threads = chunk_size / 2;
        num_blks = 1;
    }
    else
    {
        blk_threads = block_threads;
        num_blks = (chunk_size / 2) / block_threads + ((chunk_size / 2) % block_threads ? 1 : 0);
    }
    
    sum_kernel<<<num_blks, blk_threads>>>(dev_chunk, chunk_size, dev_result);
    
    int result;
    cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    check_error(cudaGetLastError(), "CUDA error occurred.");

    //printf("Partial result from process %d: %d\n", my_rank, result);
    
    return result;
}
