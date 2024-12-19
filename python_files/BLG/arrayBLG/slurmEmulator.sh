#!/bin/bash
export SLURM_CPUS_PER_TASK=1
export SLURM_ARRAY_TASK_COUNT=100
source /data1/shankar/RKKYTightBinding/.BLGvenv/bin/activate  #path to venv installed using requirements.txt
#
LOGS=logs
mkdir -p $LOGS
#
for SLURM_ARRAY_TASK_ID in {10}
do 
	echo "Starting task $SLURM_ARRAY_TASK_ID"
	python3.10 grid_calculation.py > $LOGS/L_$SLURM_ARRAY_TASK_ID.out 2> $LOGS/L_$SLURM_ARRAY_TASK_ID.err 
	echo "Task ID $SLURM_ARRAY_TASK_ID completed"
done
#
#completetion message 
echo "Slurm emulator script ended"
