#!/bin/bash

test $# -eq 1 || { echo "usage ./compile.sh N_PROC" >&2; exit; } 

N_PROC=$1

mkdir -p 1D/seq/
mkdir -p 1D/omp/
mkdir -p 1D/mpi/
mkdir -p 1D/stockham/
mkdir -p 1D/CUDA/


mkdir -p 2D/seq/
mkdir -p 2D/omp/
mkdir -p 2D/mpi/
mkdir -p 2D/mpiomp/

mkdir -p 3D/seq/
mkdir -p 3D/omp/
mkdir -p 3D/mpi/


for i in {1..5}; do
		# speedup
		#1D
		python3 plot_time.py test_sequential.out cooley-tuckey fftw 1 1D/seq/seq$i.png
		python3 plot_time.py test_stockham.out stockham fftw 1 1D/stockham/stockham$i.png
		python3 plot_time.py test_OMP.out OMP sequential_CT 1 1D/omp/omp$i.png
		python3 plot_time_MPI.py test_MPI.out $N_PROC MPI sequential_CT 1 1D/mpi/mpi$i.png

		#2D
		python3 plot_time.py test_sequential_2D.out cooley-tuckey fftw 2 2D/seq/seq$i.png
		python3 plot_time.py test_OMP_2D.out OMP sequential_CT 2 2D/omp/omp$i.png
		#python3 plot_time_MPI.py test_MPI_OMP_2D.out MPI_OMP sequential_CT 2 2D/mpiomp/mpiomp$i.png
		python3 plot_time_MPI.py test_MPI_2D.out $N_PROC MPI sequential_CT 2 2D/mpi/mpi$i.png
		
		#3D
		python3 plot_time.py test_sequential_3D.out cooley-tuckey fftw 3 3D/seq/seq$i.png
		python3 plot_time.py test_OMP_3D.out OMP sequential_CT 3 3D/omp/omp$i.png
		python3 plot_time_MPI.py test_MPI_3D.out $N_PROC MPI sequential_CT 3 3D/mpi/mpi$i.png


		#efficiency
		#1D
		SIZE=24
		python3 plot_efficiency.py test_OMP.out $N_PROC 1 1D/omp/e_omp$i.png $SIZE
		python3 plot_efficiency.py test_MPI.out $N_PROC 1 1D/mpi/e_mpi$i.png $SIZE

		#2D
		SIZE=12
		python3 plot_efficiency.py test_OMP_2D.out $N_PROC 2 2D/omp/e_omp$i.png $SIZE $SIZE
		#python3 plot_efficiency.py test_MPI_OMP_2D.out MPI_OMP sequential_CT 2 2D/mpiomp/mpiomp$i.png
		python3 plot_efficiency.py test_MPI_2D.out $N_PROC 2 2D/mpi/e_mpi$i.png $SIZE $SIZE
		
		#3D
		SIZE=8
		python3 plot_efficiency.py test_OMP_3D.out $N_PROC 3 3D/omp/e_omp$i.png $SIZE $SIZE $SIZE
		python3 plot_efficiency.py test_MPI_3D.out $N_PROC 3 3D/mpi/e_mpi$i.png $SIZE $SIZE $SIZE
done
