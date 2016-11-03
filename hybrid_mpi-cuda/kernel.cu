#include "kernel.h"

__global__ void sum_kernel(int *chunk, int chunk_size, int *result)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = chunk_size / 2; i > 0; i /= 2)
    {
        if (id < i)
        {
            chunk[id] += chunk[id + i];
        }
        __syncthreads();
    }

    if (!id)
    {
        *result = chunk[0];
    }
}
