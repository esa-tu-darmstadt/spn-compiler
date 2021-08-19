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

PARAMETERS_FILE=$1
LOG_DIR=$2
LOG_PREFIX=$3

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

echo "Sourcing virtual environment"
cd /net/celebdil/csv/spnc-static/
source venv/bin/activate

echo "Running Python script"
time python -u /net/bifur/csv/spn-compiler-eval/python-interface/test/slurm/invoke.py --logDir ${LOG_DIR} --modelName ${SPN_NAME} --modelFile ${SPN_MODEL_FILE} --inputFile ${SPN_INPUT_FILE} --referenceFile ${SPN_REFERENCE_FILE} --kernelDir ${KERNEL_DIR} --removeKernel ${REMOVE_KERNELS} --vectorize ${VECTORIZE} --vecLib ${VECTOR_LIBRARY} --shuffle ${SHUFFLE} --maxAttempts ${MAX_ATTEMPTS} --maxSuccessfulIterations ${MAX_SUCCESSFUL_ITERATIONS} --maxNodeSize ${MAX_NODE_SIZE} --maxLookAhead ${MAX_LOOK_AHEAD} --reorderInstructionsDFS ${REORDER_DFS} --allowDuplicateElements ${ALLOW_DUPLICATES} --allowTopologicalMixing ${ALLOW_TOPOLOGICAL_MIXING} &> /net/celebdil/csv/eval/slurm-logs/kernels/${LOG_PREFIX}_spn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

echo "Done"
