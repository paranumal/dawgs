#!/bin/bash

#set -x #echo on

##################################################
# Description
##################################################
# This script scales the dawgs performance tests
# on several MPI ranks. The problem size is kept
# equal on ranks and increased every iteration.


# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "Weak Scaling helper script"
  echo "./weak_scale.sh "
  echo "    [-n]           Number of MPI ranks to use.                "
  echo "    [-N]           Number of nodes to run on (used for srun). "
  echo "    [-p]           Polynomial degree of mesh.                 "
  echo "    [-m|--mode]    Runtime mode (Serial, OpenMP, CUDA, HIP,   "
  echo "                   or OpenCL).                                "
  echo "    [--ga]         Run GPU-aware MPI tests (Default: false).  "
  echo "    [-v|--verbose] Print MPI rank binding (Default: false).   "
  echo "    [-h|--help]    Prints this help message.                  "
}

# #################################################
# global variables
# #################################################
verbose=false
gpu_aware=false
N=1
n=1
p=4
mode="HIP" #default to HIP mode

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,verbose,ga,mode --options hvn:N:m:p: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -v|--verbose)
        verbose=true
        shift ;;
    --ga)
        gpu_aware=true
        shift ;;
    -n)
        n=${2}
        shift 2 ;;
    -N)
        N=${2}
        shift 2 ;;
    -m|--mode)
        mode=${2}
        shift 2 ;;
    -p)
        p=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

#check mode
if ! [[ "$mode" =~ ^(Serial|OpenMP|CUDA|HIP|OpenCL)$ ]]; then
  echo "Mode must be one of: Serial, OpenMP, CUDA, HIP, or OpenCL";
  exit 1
fi

S_low=2
S_step=2
S_high=20
if [[ $p -eq 1 ]]; then
  S_low=2
  S_step=8
  S_high=122
elif [[ $p -eq 2 ]]; then
  S_low=2
  S_step=4
  S_high=102
elif [[ $p -eq 3 ]]; then
  S_low=2
  S_step=4
  S_high=82
elif [[ $p -eq 4 ]]; then
  S_low=2
  S_step=4
  S_high=62
elif [[ $p -eq 5 ]]; then
  S_low=2
  S_step=4
  S_high=54
elif [[ $p -eq 6 ]]; then
  S_low=2
  S_step=2
  S_high=38
elif [[ $p -eq 7 ]]; then
  S_low=2
  S_step=2
  S_high=28
elif [[ $p -eq 8 ]]; then
  S_low=2
  S_step=2
  S_high=28
fi



# #################################################
# Run
# #################################################

#options for MPI
# mpi_opts="-mca pml ucx -mca btl ^uct"

dawgs_opts="-m ${mode} -p ${p}"
if [[ "${gpu_aware}" == true ]]; then
  dawgs_opts="-ga ${dawgs_opts}"
fi
if [[ "${verbose}" == true ]]; then
  dawgs_opts="-v ${dawgs_opts}"
fi

make -j `nproc`

for S in `seq ${S_low} ${S_step} ${S_high}`
do
    #echo "Problem size:" $S"x"$S"x"$S

    #mpi run
    mpirun -np $n ${mpi_opts} dawgsMain -nx $S -ny $S -nz $S ${dawgs_opts}
    
    #slurm run

    #srun -N $N -n $n -p amdMI60 dawgsMain -nx $S -ny $S -nz $S ${dawgs_opts}
done
