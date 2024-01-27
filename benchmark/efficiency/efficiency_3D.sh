#!/bin/bash

TEST=$1
SIZE1=$2
SIZE2=$3
SIZE3=$4
P_NUM=$5

echo p,SPEEDUP_F,SPEEDUP_B,EFFICIENCY_F,EFFICIENCY_B

for ((i=1; i<=$P_NUM; i=$i)); do
	if grep --quiet -i "MPI" <<< "$1" ;then 
		test $# -eq 5 || { echo "usage ./compile.sh SIZE1 SIZE2 SIZE3 processes" >&2; exit; } 
		OUT=$(mpirun --use-hwthread-cpus -n $i $TEST $SIZE1 $SIZE2 $SIZE3)
	else
		test $# -eq 5 || { echo "usage ./compile.sh SIZE1 SIZE2 SIZE3 processes" >&2 ; exit; } 
		OUT=$(OMP_NUM_THREADS=$i $TEST $SIZE1 $SIZE2 $SIZE3)
	fi
	
	SPEEDUP_F=$(cut -d',' -f 8 <<< $OUT)
	EFFICIENCY_F=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_F / $i) }")
	SPEEDUP_B=$(cut -d',' -f 9 <<< $OUT)
	EFFICIENCY_B=$(awk "BEGIN { printf(\"%.2f\", $SPEEDUP_B / $i) }")

	echo "$i,$SPEEDUP_F,$SPEEDUP_B,$EFFICIENCY_F,$EFFICIENCY_B"
	
	if grep --quiet -i "MPI" <<< "$1" ;then 
		i=$(( $i * 2))
	else
		i=$(($i + 1))
	fi
done
