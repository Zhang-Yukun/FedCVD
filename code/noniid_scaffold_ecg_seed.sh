seeds=(42)
server_lrs=(1 0.1 0.01)
client_lrs=(0.1 0.01)

for seed in ${seeds[@]}; do
  for server_lr in ${server_lrs[@]}; do
    for client_lr in ${client_lrs[@]}; do
      case_name="scaffold_ecg-server_lr=${server_lr}-client_lr=${client_lr}-seed=${seed}"
      python noniid_scaffold_ecg.py --server_lr "$server_lr" --client_lr "$client_lr" --case_name "$case_name" --seed "$seed"
    done
  done
done