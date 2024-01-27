#!/bin/bash

TEST=$1
SIZE=$2
P_NUM=$3

echo p,SPEEDUP_F,SPEEDUP_B,EFFICIENCY_F,EFFICIENCY_B

for ((i=1; i<=$P_NUM; i=i*2)); do
	test $# -eq 3 || { echo "usage ./compile.sh SIZE MAX_N_PROC" >&2; exit; } 

	if grep --quiet -i "MPI" <<< "$1" ;then 
		OUT=$(mpirun --use-hwthread-cpus -n $i $TEST $SIZE)
	else
		OUT=$(OMP_NUM_THREADS=$i $TEST $SIZE)
	fi
	
	SPEEDUP_F=$(cut -d',' -f 6 <<< $OUT)
	EFFICIENCY_F=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_F / $i) }")
	SPEEDUP_B=$(cut -d',' -f 7 <<< $OUT)
	EFFICIENCY_B=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_B / $i) }")

	echo "$i,$SPEEDUP_F,$SPEEDUP_B,$EFFICIENCY_F,$EFFICIENCY_B"
done


