#!/bin/bash

make

rm output.csv

export NX=100
export NY=100
export NZ=100

echo "Problem size:" $NX"x"$NY"x"$NZ;

for i in {1..8};\
    do j=$(mpirun --mca osc ucx -x UCX_TLS=self,posix,rocm_ipc,rocm_copy -np $i ./dawgsMain -v -m HIP -ga -nx $NX -ny $NY -nz $NZ | grep 'Time taken' | awk '{print $4}'); \
    echo $i','$j | tee -a output.csv; \
    #do mpirun --mca pml ucx --mca btl ^vader,tcp,openib,uct -np $i ./dawgsMain -v -m HIP -ga; \
done
#grep 'Time taken' | awk '{print $4}' 2>&1 | tee some.log;