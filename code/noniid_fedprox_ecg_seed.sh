seeds=(3 9 55 15)

for seed in ${seeds[@]}; do
    python noniid_fedprox_ecg.py --seed $seed
done