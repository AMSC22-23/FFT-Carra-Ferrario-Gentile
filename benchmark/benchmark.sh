#!/bin/bash

TEST=$1
PROCESSES_NUM=$2

echo "n,$3 forward,$3 inverse,$4 forward,$4 inverse,SU forward,SU inverse,error"
for n in {14..19}; do
	mpirun --use-hwthread-cpus -n $PROCESSES_NUM $TEST $n
done
