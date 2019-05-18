# !/bin/bash

for seed in {0..4}
do
    echo $seed
    python initialize_HAC.py --retrain --seed $seed
done

