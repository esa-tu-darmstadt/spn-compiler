#! /bin/bash

#SBATCH -J dfs-part
#SBATCH -a 1-897%4
#SBATCH -e /net/celebdil/csv/eval/slurm-logs/kernels_partitioned/stderr_%A_%a.out
#SBATCH -o /net/celebdil/csv/eval/slurm-logs/kernels_partitioned/stdout_%A_%a.out
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
  >&2 echo "parameter #4: the spnc source code directory"
  >&2 echo "parameter #5: the spnc build directory"
  exit 2
fi

PARAMETERS_FILE=$1
LOG_DIR=$2
LOG_PREFIX=$3
SRC_DIR=$4
BUILD_DIR=$5

NUM_COLUMNS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print NF}}" ${PARAMETERS_FILE})
if [[ ${NUM_COLUMNS} -ne 17 ]]; then
  >&2 echo "expected 17 columns in parameter file at line ${SLURM_ARRAY_TASK_ID}, but got ${NUM_COLUMNS}"
  exit 2
fi

SPN_NAME=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$1}}" ${PARAMETERS_FILE})
SPN_MODEL_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$2}}" ${PARAMETERS_FILE})
SPN_INPUT_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$3}}" ${PARAMETERS_FILE})
SPN_REFERENCE_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$4}}" ${PARAMETERS_FILE})
KERNEL_DIR=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$5}}" ${PARAMETERS_FILE})
REMOVE_KERNELS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$6}}" ${PARAMETERS_FILE})
VECTORIZE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$7}}" ${PARAMETERS_FILE})
VECTOR_LIBRARY=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$8}}" ${PARAMETERS_FILE})
SHUFFLE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$9}}" ${PARAMETERS_FILE})
MAX_ATTEMPTS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$10}}" ${PARAMETERS_FILE})
MAX_SUCCESSFUL_ITERATIONS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$11}}" ${PARAMETERS_FILE})
MAX_NODE_SIZE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$12}}" ${PARAMETERS_FILE})
MAX_LOOK_AHEAD=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$13}}" ${PARAMETERS_FILE})
REORDER_DFS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$14}}" ${PARAMETERS_FILE})
ALLOW_DUPLICATES=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$15}}" ${PARAMETERS_FILE})
ALLOW_TOPOLOGICAL_MIXING=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$16}}" ${PARAMETERS_FILE})
MAX_TASK_SIZE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$17}}" ${PARAMETERS_FILE})

echo "Sourcing virtual environment"
cd ${BUILD_DIR}
source venv/bin/activate

echo "Running Python script"
time python -u ${SRC_DIR}/python-interface/test/slurm/invoke.py --logDir ${LOG_DIR} --modelName ${SPN_NAME} --modelFile ${SPN_MODEL_FILE} --inputFile ${SPN_INPUT_FILE} --referenceFile ${SPN_REFERENCE_FILE} --kernelDir ${KERNEL_DIR} --removeKernel ${REMOVE_KERNELS} --vectorize ${VECTORIZE} --vecLib ${VECTOR_LIBRARY} --shuffle ${SHUFFLE} --maxAttempts ${MAX_ATTEMPTS} --maxSuccessfulIterations ${MAX_SUCCESSFUL_ITERATIONS} --maxNodeSize ${MAX_NODE_SIZE} --maxLookAhead ${MAX_LOOK_AHEAD} --reorderInstructionsDFS ${REORDER_DFS} --allowDuplicateElements ${ALLOW_DUPLICATES} --allowTopologicalMixing ${ALLOW_TOPOLOGICAL_MIXING} --maxTaskSize ${MAX_TASK_SIZE} &> /net/celebdil/csv/eval/slurm-logs/kernels_partitioned/${LOG_PREFIX}_spn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

echo "Done"
