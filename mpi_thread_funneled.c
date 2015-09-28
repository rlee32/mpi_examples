// Author: Robert H. Lee
// Created: Fall 2015
// The motivation for this example is utilization of MPI_THREAD_FUNNELED.
// This allows a normally blocking MPI command to run in the background 
// while other threads can process work in parallel. 
// MPI_THREAD_FUNNELED means that all MPI calls are sent to the main thread, 
// (and serialized), even if MPI calls are invoked by other threads. 
// Therefore we want other threads to be doing the main work, and then any 
// other thread can make the MPI calls (which will all be funneled to the main 
// thread).

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>

#include "timer.h"

// Here we execute a for loop, where we want data associated with each loop
// to be transmitted by an MPI process, but we are able to process the entire 
// for loop all at once. So, instead of waiting on the blocking MPI call at the 
// end of each loop (or even using an MPI call to handle all data all at once 
// at the end of the whole for loop), we can transmit the data as it is done in 
// each loop, and continue working on the rest of the data. The idea is to 
// achieve high overlap of communication and computation. 

int main(int argc, char** argv)
{
  // Initialize MPI threads and check provided support levels.
  int provided_support = -1;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided_support);
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
  int size_per_processor = 1e8;
  int total_size = processors*size_per_processor;
  int *global_data = (int*) malloc(total_size*sizeof(int));
  int offset = size_per_processor*rank;
  int *local_data = global_data + offset;
  int i = 0;
  for(; i < size_per_processor; ++i) local_data[i] = i+offset;

  // clock_t start_t = clock();

  // Initiate the parallel region so that multiple threads can be used.
  #pragma omp parallel
  {
    // We do not want the computational work on the main thread; we only want 
    // the main thread to handle the MPI communication. The master thread has 
    // omp_get_thread_num() == 0.
    if(omp_get_thread_num() != 0)
    {
      // Now we only want one non-master thread to do work, hence 'single'. If 
      // we did not use single, redundant, same computations will be done.
      // The 'nowait' allows us to continue work without waiting on other 
      // things inside the loop (in this case, 'task').
      #pragma omp single nowait
      {
        tic();
        for(int i = 0; i < 1; ++i)
        {
          // Our 'work'.
          sleep(1);
          fprintf(stdout, "Work %d, thread: %d, rank %d\n", i, 
            omp_get_thread_num(), rank);
          
          // This would be where a blocking call would be placed. Placed inside 
          // the task, the call would be placed upon the master thread.
          // 'task' gives up the code inside to the general available thread 
          // pool.
          // Calling this task after the 'work' above guarantees that the 
          // info generated above will be complete when handled by MPI calls 
          // inside this task.
          // if(omp_get_thread_num() == rank)
          #pragma omp task
          {
            // A blocking call, such as MPI_Allgatherv().
            // sleep(1);
            fprintf(stdout, "MPI call %d, thread: %d, rank %d\n", i, 
              omp_get_thread_num(), rank);
            tic();
            MPI_Allgather(
              local_data, 
              size_per_processor, 
              MPI_INT, 
              global_data, 
              size_per_processor, 
              MPI_INT, 
              MPI_COMM_WORLD
            );
            double timing = toc();
            fprintf(stdout,"allgather: %f\n", timing);
          }
        }
        double timing = toc();
        fprintf(stdout,"loop time: %f\n", timing);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // clock_t end_t = clock();
  // double total_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
  // fprintf(stdout,"overall: %f\n", total_time);

  free(global_data);
  MPI_Finalize();
  return EXIT_SUCCESS;
}