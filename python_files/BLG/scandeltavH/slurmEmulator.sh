#!/bin/bash
export SLURM_CPUS_PER_TASK=6
source /data1/shankar/RKKYTightBinding/.BLGvenv/bin/activate  #path to venv installed using requirements.txt
#
LOGS=logs/localrun/
mkdir -p $LOGS
#
python3.10 scandelta.py > $LOGS/delta_$SLURM_ARRAY_TASK_ID.out 2> $LOGS/delta_$SLURM_ARRAY_TASK_ID.err 
#
#completetion message 
echo "Slurm emulator script ended"
