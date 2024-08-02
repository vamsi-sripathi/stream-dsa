#!/bin/bash

export KMP_AFFINITY=granularity=fine,compact,1,0

sz=$((1024*1024*1024))

dev=cpu

for t in {1..30}; 
do
  export OMP_NUM_THREADS=$t
  for ker in triad dot
  do
     if [[ "$ker" == "scale" || "$ker" == "triad" ]]; then
        suff=_nt
     else
        suff=""
     fi

     echo "Running $ker with $t threads"          2>&1 | tee -a ${dev}_${ker}.log
     numactl -m0 ./${dev}_${ker}${suff}.bin $sz   2>&1 | tee -a ${dev}_${ker}.log
  done
done
