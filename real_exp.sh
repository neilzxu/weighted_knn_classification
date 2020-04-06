#!/usr/bin/env bash


ks=$(seq 120 20 200)

source activate rakuten

python train.py real

for k in $ks; do
   python train.py grid --k $k
done
