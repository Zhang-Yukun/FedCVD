seeds=(3 9 55 15)

for seed in ${seeds[@]}; do
    python scaffold_ecg_seed.py --seed $seed
done