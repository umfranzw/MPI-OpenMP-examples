#ifndef _CRUNCH_H
#define _CRUNCH_H

void check_error(cudaError_t status, const char *msg);
int get_max_block_threads();
int crunch(int my_rank, int *chunk, int chunk_size);

#endif
