#!/bin/bash
ARGS=`head -${SLURM_ARRAY_TASK_ID} arguments.txt | tail -1`

python -u tonic_classifier.py --num-epochs $1 $ARGS

for E in $(seq 0 $1)
do
    python -u tonic_classifier_evaluate.py --trained-epoch $E $ARGS
done
