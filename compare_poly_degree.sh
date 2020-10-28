#!/bin/bash

make

rm output.csv

export NPROC=4;

export NX=150
export NY=150
export NZ=150

export GA=1
# 1: Enabled; 0: Disabled;

echo "Problem size:" $NX"x"$NY"x"$NZ;
if [[ $GA -eq 1 ]]
then
    echo "GPU-Aware: Enabled";
elif [[ $GA -eq 0 ]]
then
    echo "GPU-Aware: Disabled";
fi

for i in {1..7};\
    do echo " "; \
    if [[ $GA -eq 1 ]]
    then
        echo "mpirun -np "$NPROC" -mca pml ucx -mca btl ^uct dawgsMain -nx "$NX" -ny "$NY" -nz "$NZ" -p "$i" -v -m HIP -ga"; \
        j=$(mpirun -np $NPROC -mca pml ucx -mca btl ^uct dawgsMain -nx $NX -ny $NY -nz $NZ -p $i -v -m HIP -ga 2>/dev/null | grep -E 'Error|Time taken'); \
    elif [[ $GA -eq 0 ]]
    then
        echo "mpirun -np "$NPROC" -mca pml ucx -mca btl ^uct dawgsMain -nx "$NX" -ny "$NY" -nz "$NZ" -p "$i" -v -m HIP"; \
        j=$(mpirun -np $NPROC -mca pml ucx -mca btl ^uct dawgsMain -nx $NX -ny $NY -nz $NZ -p $i -v -m HIP 2>/dev/null | grep -E 'Error|Time taken'); \
    fi
    #do j=$(mpirun --mca osc ucx -x UCX_TLS=self,posix,rocm_ipc,rocm_copy -np $i ./dawgsMain -v -m HIP -ga -nx $NX -ny $NY -nz $NZ | grep 'Time taken' | awk '{print $4}'); \
    echo $j | grep 'Error' | awk '{printf $1" "$2" "$3"\n"}';\
    k=$(echo $j | grep 'Error' | awk '{print $7}')
    echo $i','$k | tee -a output.csv; \
    #do mpirun --mca pml ucx --mca btl ^vader,tcp,openib,uct -np $i ./dawgsMain -v -m HIP -ga; \
done
#grep 'Time taken' | awk '{print $4}' 2>&1 | tee some.log;