This repository contains use cases and examples of MPI.

# Compilation
mpicc -fopenmp -std=gnu99 <source_file>

# Source File Dscriptions
thread_funneled.c: Here we overlap computation and communication using 
MPI_THREAD_FUNNELED. To manage which threads are calling MPI and which are 
performing work, we use omp pragmas and keywords.

allgather_test.c: Here we test how the input arguments to MPI_Allgather() can 
be used; specifically, using it to allgather at locations that correspond to 
block divisions, but not filling up the whole block (see source for details).
