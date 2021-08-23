#! /bin/bash

#SBATCH -J spnc-dfs
#SBATCH -a 1-897%4
#SBATCH -e /net/celebdil/csv/eval/slurm-logs/kernels/stderr_%A_%a.out
#SBATCH -o /net/celebdil/csv/eval/slurm-logs/kernels/stdout_%A_%a.out
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --partition=Epyc
#SBATCH --mem-per-cpu=4096
#SBATCH -t 168:00:00

if [[ $# -ne 5 ]]; then
  >&2 echo "Error: missing arguments"
  >&2 echo "parameter #1: the array file with python parameters"
  >&2 echo "parameter #2: the directory to write python data to (where csv data will be stored, ...)"
  >&2 echo "parameter #3: the prefix to attach to log files"
  >&2 echo "parameter #4: the spnc source code directory containing the python script"
  >&2 echo "parameter #5: the spnc build directory"
  exit 2
fi

PARAMETERS_FILE=$1
LOG_DIR=$2
LOG_PREFIX=$3
SRC_DIR=$4
BUILD_DIR=$5

ARGUMENTS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print}}" ${PARAMETERS_FILE})

echo "Sourcing virtual environment"
cd ${BUILD_DIR}
source venv/bin/activate

echo "Running Python script"
time python -u ${SRC_DIR}/python-interface/test/slurm/invoke.py --logDir ${LOG_DIR} ${ARGUMENTS} &> /net/celebdil/csv/eval/slurm-logs/${LOG_PREFIX}_spn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

echo "Done"
