#!/bin/bash

TEST_OMP=$1
echo "n,$2 forward,$2 inverse,$3 forward,$3 inverse,SU forward,SU inverse,error"
for n in {14..19}; do
	$TEST_OMP $n
done
