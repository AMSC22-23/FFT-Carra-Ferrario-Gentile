#!/bin/bash

TEST=$1
PROCESSES_NUM=$2

if grep --quiet -i "MPI" <<< "$1" ;then 
	echo "n,$3 forward,$4 forward,$3 inverse,$4 inverse,SU forward,SU inverse,error forward, error inverse"
else
	echo "n,$2 forward,$3 forward,$2 inverse,$3 inverse,SU forward,SU inverse,error forward, error inverse"
fi
for n in {3..8}; do	
	if grep --quiet -i "MPI" <<< "$1" ;then 
		test $# -eq 4 || { echo "usage ./benchmark3D.sh test.out MPI_processes strategy_header_1 strategy_header_2" >&2; exit; } 
		echo  $((8 + 8 + $n)),$(mpirun --use-hwthread-cpus -n $PROCESSES_NUM $TEST 8 8 $n | cut -f 4- -d',')
	else
		test $# -eq 3 || { echo "usage ./benchmark3D.sh test.out strategy_header_1 strategy_header_2" >&2 ; exit; } 
		echo  $((8 + 8 + $n)),$($TEST 8 8 $n | cut -f 4- -d',')
	fi
done


