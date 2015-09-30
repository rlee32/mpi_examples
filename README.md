This repository contains use cases and examples of MPI.

# Compilation
mpicc -fopenmp -std=gnu99 <source_files>

# Source File Dscriptions
mpi_thread_funneled.c: Here we overlap computation and communication using 
MPI_THREAD_FUNNELED. To manage which threads are calling MPI and which are 
performing work, we use omp pragmas and keywords.

