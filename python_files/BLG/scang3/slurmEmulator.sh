#!/bin/bash
export SLURM_CPUS_PER_TASK=6
source /data1/shankar/RKKYTightBinding/.BLGvenv/bin/activate  #path to venv installed using requirements.txt
#
LOGS=logs/turnoffg4
mkdir -p $LOGS
#
for SLURM_ARRAY_TASK_ID in {0..12}
do 
	echo "Starting task $SLURM_ARRAY_TASK_ID"
	python3.10 scang3.py $SLURM_ARRAY_TASK_ID > $LOGS/g3_$SLURM_ARRAY_TASK_ID.out 2> $LOGS/g3_$SLURM_ARRAY_TASK_ID.err 
	echo "Task ID $SLURM_ARRAY_TASK_ID completed"
done
#
#completetion message 
echo "Slurm emulator script ended"
