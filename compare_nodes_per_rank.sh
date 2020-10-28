#!/bin/bash

make

rm output.csv

export NX=140
export NY=140
export NZ=140

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

for i in {3..8};\
    do echo " "; \
    if [[ $GA -eq 1 ]]
    then
        echo "mpirun -np "$i" -mca pml ucx -mca btl ^uct dawgsMain -nx "$NX" -ny "$NY" -nz "$NZ" -v -m HIP -ga"; \
        j=$(mpirun -np $i -mca pml ucx -mca btl ^uct dawgsMain -nx $NX -ny $NY -nz $NZ -v -m HIP -ga 2>/dev/null | grep -E 'Error|Time taken|Global nodes'); \
    elif [[ $GA -eq 0 ]]
    then
        echo "mpirun -np "$i" -mca pml ucx -mca btl ^uct dawgsMain -nx "$NX" -ny "$NY" -nz "$NZ" -v -m HIP"; \
        j=$(mpirun -np $i -mca pml ucx -mca btl ^uct dawgsMain -nx $NX -ny $NY -nz $NZ -v -m HIP 2>/dev/null | grep -E 'Error|Time taken|Global nodes'); \
    fi
    #do j=$(mpirun --mca osc ucx -x UCX_TLS=self,posix,rocm_ipc,rocm_copy -np $i ./dawgsMain -v -m HIP -ga -nx $NX -ny $NY -nz $NZ | grep 'Time taken' | awk '{print $4}'); \
    echo $j | grep 'Error' | awk '{printf $7" "$8" "$9"\n"}';\
    k=$(echo $j | grep 'Global' | awk '{print $6}');\
    l=$(echo $j | grep 'Time' | awk '{print $13}');\
    echo $i','$k','$l | tee -a output.csv; \
    #do mpirun --mca pml ucx --mca btl ^vader,tcp,openib,uct -np $i ./dawgsMain -v -m HIP -ga; \
done
#grep 'Time taken' | awk '{print $4}' 2>&1 | tee some.log;