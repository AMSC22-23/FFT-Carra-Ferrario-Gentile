#!/bin/bash

TEST=$1
PROCESSES_NUM=$2

if grep --quiet -i "MPI" <<< "$1" ;then 
	echo "n,$3 m,$3 k,$3 forward,$3 inverse,$4 forward,$4 inverse,SU forward,SU inverse,error"
else
	echo "n,$2 m,$2 k,$3 forward,$2 inverse,$3 forward,$3 inverse,SU forward,SU inverse,error"
fi
for n in {8..9}; do
	for m in {8..9}; do
		for k in {8..9}; do	
			if grep --quiet -i "MPI" <<< "$1" ;then 
				test $# -eq 4 || { echo "usage ./benchmark.sh test.out MPI_processes strategy_header_1 strategy_header_2" >&2; exit; } 
				mpirun --use-hwthread-cpus -n $PROCESSES_NUM $TEST $n $m $k
			else
				test $# -eq 3 || { echo "usage ./benchmark.sh test.out strategy_header_1 strategy_header_2" >&2 ; exit; } 
				$TEST $n $m $k
			fi
		done
	done
done


