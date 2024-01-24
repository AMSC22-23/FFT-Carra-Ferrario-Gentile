#!/bin/bash

TEST=$1
SIZE=$2
P_NUM=$3

#if grep --quiet -i "MPI" <<< "$1" ;then 
#	echo "n,$3 forward,$3 inverse,$4 forward,$4 inverse,SU forward,SU inverse,error"
#else
#	echo "n,$2 forward,$2 inverse,$3 forward,$3 inverse,SU forward,SU inverse,error"
#fi

for ((i=1; i<=$P_NUM; i=i*2)); do
	if grep --quiet -i "MPI" <<< "$1" ;then 
		test $# -eq 5 || { echo "usage ./compile.sh SIZE MPI_processes strategy_header_1 strategy_header_2" >&2; exit; } 
		OUT=$(mpirun --use-hwthread-cpus -n $i $TEST $SIZE)
	else
		test $# -eq 5 || { echo "usage ./compile.sh SIZE strategy_header_1 strategy_header_2" >&2 ; exit; } 
		OUT=$(OMP_NUM_THREADS=$i $TEST $SIZE)
	fi
	
	SPEEDUP_F=$(cut -d',' -f 6 <<< $OUT)
	EFFICIENCY_F=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_F / $i) }")
	SPEEDUP_B=$(cut -d',' -f 7 <<< $OUT)
	EFFICIENCY_B=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_B / $i) }")

	echo "N_PROC $i: Speedup forward $SPEEDUP_F, efficiency $EFFICIENCY_F"
	echo "N_PROC $i: Speedup forward $SPEEDUP_B, efficiency $EFFICIENCY_B"
done


