#!/bin/bash
export SLURM_CPUS_PER_TASK=6
source .BLG #path to venv installed using requirements.txt
#
mkdir -p logs 
#
for $SLURM_ARRAY_TASK_ID in $(seq 0 20)
do 
	echo "Starting task $SLURM_ARRAY_TASK_ID"
	python blgNLDOS.py $SLURM_ARRAY_TASK_ID
	echo "Task ID $SLURM_ARRAY_TASK_ID completed"
done
#
#completetion message 
echo "Slurm emulator script ended"
