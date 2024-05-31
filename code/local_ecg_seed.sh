client_ids=(1 2 3 4)
seeds=(3 9 55 15)

for client_id in ${client_ids[@]}; do
    for seed in ${seeds[@]}; do
        python local_ecg_seed.py --client_id $client_id --seed $seed
    done
done