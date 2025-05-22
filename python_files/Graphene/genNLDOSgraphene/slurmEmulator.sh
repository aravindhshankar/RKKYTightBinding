#!/bin/bash
export SLURM_CPUS_PER_TASK=1
# export SLURM_ARRAY_TASK_COUNT=100
export SLURM_ARRAY_TASK_COUNT=1
# source /data1/shankar/RKKYTightBinding/.BLGvenv/bin/activate  #path to venv installed using requirements.txt
# source /scratch/ashankar/RKKYTightBinding/.BLGvenv/bin/activate  #path to venv installed using requirements.txt
source ~/Documents/GitHub/RKKYTightBinding/.BLGvenv/bin/activate

#
LOGS=logs
mkdir -p $LOGS
#
# for SLURM_ARRAY_TASK_ID in {10,}
for SLURM_ARRAY_TASK_ID in {0,}
do 
	echo "Starting task $SLURM_ARRAY_TASK_ID"
	export SLURM_ARRAY_TASK_ID
	python3.10 grid-calculation.py > $LOGS/L_$SLURM_ARRAY_TASK_ID.out 2> $LOGS/L_$SLURM_ARRAY_TASK_ID.err 
	echo "Task ID $SLURM_ARRAY_TASK_ID completed"
done
#
#python3.10 results_combiner.py
#completetion message 
echo "Slurm emulator script ended"
