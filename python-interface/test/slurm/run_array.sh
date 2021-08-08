#! /bin/bash

#SBATCH -J spnc-eval
#SBATCH -a 1-897%12
#SBATCH -e /net/celebdil/csv/eval/slurm-logs/stderr_%A_%a.out
#SBATCH -o /net/celebdil/csv/eval/slurm-logs/stdout_%A_%a.out
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --partition=Epyc
#SBATCH --mem-per-cpu=8192
#SBATCH -t 96:00:00

PARAMETERS_FILE=$1
LOG_DIR=$2

# modelName modelFile inputFile referenceFile vectorize vecLib shuffle maxAttempts maxSuccessfulIterations maxNodeSize maxLookAhead reorderInstructionsDFS allowDuplicateElements allowTopologicalMixing

SPN_NAME=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$1}}" ${PARAMETERS_FILE})
SPN_MODEL_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$2}}" ${PARAMETERS_FILE})
SPN_INPUT_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$3}}" ${PARAMETERS_FILE})
SPN_REFERENCE_FILE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$4}}" ${PARAMETERS_FILE})
VECTORIZE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$5}}" ${PARAMETERS_FILE})
VECTOR_LIBRARY=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$6}}" ${PARAMETERS_FILE})
SHUFFLE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$7}}" ${PARAMETERS_FILE})
MAX_ATTEMPTS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$8}}" ${PARAMETERS_FILE})
MAX_SUCCESSFUL_ITERATIONS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$9}}" ${PARAMETERS_FILE})
MAX_NODE_SIZE=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$10}}" ${PARAMETERS_FILE})
MAX_LOOK_AHEAD=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$11}}" ${PARAMETERS_FILE})
REORDER_DFS=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$12}}" ${PARAMETERS_FILE})
ALLOW_DUPLICATES=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$13}}" ${PARAMETERS_FILE})
ALLOW_TOPOLOGICAL_MIXING=$(awk "{if (NR==${SLURM_ARRAY_TASK_ID}) {print \$14}}" ${PARAMETERS_FILE})

echo "Sourcing virtual environment"
cd /net/celebdil/csv/spnc-dynamic/
source venv/bin/activate

echo "Running Python script"
#python -v /net/bifur/csv/spn-compiler-eval/python-interface/test/slurm/cpuExecutionSlurm.py nltcs_100_200_4_3_3_3_1.0_True_spn_0 /net/celebdil/spn-benchmarks/ratspn-classification/models/nltcs_100_200_4_3_3_3_1.0_True/spn_0.bin /net/celebdil/spn-benchmarks/ratspn-classification/io_files/nltcs/input.csv /net/celebdil/spn-benchmarks/ratspn-classification/io_files/nltcs/nltcs_100_200_4_3_3_3_1.0_True_spn_0.csv True None False 1 1 10000 3 True False False &> /net/celebdil/csv/eval/slurm-logs/param-spn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
time python -u /net/bifur/csv/spn-compiler-eval/python-interface/test/slurm/invoke.py ${LOG_DIR} ${SPN_NAME} ${SPN_MODEL_FILE} ${SPN_INPUT_FILE} ${SPN_REFERENCE_FILE} ${VECTORIZE} ${VECTOR_LIBRARY} ${SHUFFLE} ${MAX_ATTEMPTS} ${MAX_SUCCESSFUL_ITERATIONS} ${MAX_NODE_SIZE} ${MAX_LOOK_AHEAD} ${REORDER_DFS} ${ALLOW_DUPLICATES} ${ALLOW_TOPOLOGICAL_MIXING} &> /net/celebdil/csv/eval/slurm-logs/param-spn_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

echo "Done"
