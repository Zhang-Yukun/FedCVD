communication_round=25
options=('adam' 'yogi' 'adagrad')
server_lrs=(0.1 0.0316227 0.001)
client_lrs=(0.1 0.001)
beta1s=(0.9)
beta2s=(0.999)
taus=(1e-2)

for opt in "${options[@]}"; do
  for server_lr in "${server_lrs[@]}"; do
    for client_lr in "${client_lrs[@]}"; do
      for beta1 in "${beta1s[@]}"; do
        for beta2 in "${beta2s[@]}"; do
          for tau in "${taus[@]}"; do
            case_name="${opt}-server_lr=${server_lr}-client_lr=${client_lr}-beta1=${beta1}-beta2=${beta2}-tau=${tau}"
            echo "Running BiSR with case_name=$case_name"
            python ./fedopt_ecg.py \
              --case_name "$case_name" \
              --communication_round "$communication_round" \
              --option "$opt" \
              --server_lr "$server_lr" \
              --client_lr "$client_lr" \
              --beta1 "$beta1" \
              --beta2 "$beta2" \
              --tau "$tau"
          done
        done
      done
    done
  done
done
