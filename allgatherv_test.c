// Author: Robert H. Lee
// Created: Fall 2015
// Here we use allgatherv to fill only part of the blocks corresponding to each 
// process. This is useful when we want to overlap computation and 
// communication. If computation is required to produce the block of data that 
// needs to be allgathered, we can compute a small part of the block, then 
// initiate allgatherv on the small part, then continue computation and 
// allgathering successive chunks. The allgatherv and computation can be 
// performed in parallel by the use of threads.

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>

int main(int argc, char** argv)
{
  // Initialize MPI threads and check provided support levels.
  int provided_support = -1;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_support);
  int processors = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &processors);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0)
  {
    fprintf(stdout, "Total processors, rank: %d, %d\n", processors, rank);
    fprintf(stdout, "Support levels: %d, %d, %d, %d\n", MPI_THREAD_SINGLE, 
      MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
    fprintf(stdout, "Requested, provided support: %d, %d\n", 
      MPI_THREAD_FUNNELED, provided_support);
  }

  // Setup allgather environment.
  int steps = 3;
  int size_per_step = 5;
  int size_per_processor = size_per_step * steps;
  int total_size = processors*size_per_processor;
  int *global_data = (int*) malloc(total_size*sizeof(int));
  int i = 0;
  for(i = 0; i < total_size; ++i) global_data[i] = -1;
  int offset = size_per_processor*rank;
  int *local_data = global_data + offset;
  for(i = 0; i < size_per_processor; ++i) local_data[i] = i+offset;
  
  // Compute displs and recvcounts.
  int* displs = (int*) malloc(processors*sizeof(int));
  int* recvcounts = (int*) malloc(processors*sizeof(int));
  for(i = 0; i < processors; ++i)
  {
    displs[i] = i * size_per_processor;
    recvcounts[i] = size_per_step;
  }

  // If the send size differs, allgather hangs.
  for(i = 0; i < steps; ++i)
  {
    int block = size_per_step;//(rank & 1) ? 0: size_per_step;
    fprintf(stdout, "Rank: %d, send count: %d\n", rank, block);
    MPI_Allgatherv(
      local_data, 
      block, 
      MPI_INT, 
      global_data, 
      recvcounts, // replaces recvcount
      displs, // addition for allgatherv
      MPI_INT, 
      MPI_COMM_WORLD
    );
    // update locations
    local_data += size_per_step;
    int j;
    for(j = 0; j < processors; ++j) displs[j] += size_per_step;
  }


  // Print list.
  fprintf(stdout, "Rank %d:\n", rank);
  for(i = 0; i < total_size; ++i)
  {
    fprintf(stdout, "%d ", global_data[i]);
  }
  fprintf(stdout, "\n");

  // Clean up.
  free(global_data);
  free(displs);
  free(recvcounts);

  MPI_Finalize();

  return EXIT_SUCCESS;
}