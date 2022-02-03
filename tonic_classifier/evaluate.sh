#!/bin/bash
for E in $(seq 0 $1)
do
    python tonic_classifier_evaluate.py --trained-epoch $E "${@:2}"
done
