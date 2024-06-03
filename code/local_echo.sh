seeds=(3 9 15 55)
models=("unet")
clients=(1 2 3)
batch_sizes=(32)
lrs=(0.1)

for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
    for client in "${clients[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for lr in "${lrs[@]}"; do
          case_name="client${client}-model=${model}-batch_size=${batch_size}-lr=${lr}-seed=${seed}"
          python local_echo.py \
            --case_name "$case_name" \
            --batch_size "$batch_size" \
            --model "$model" \
            --lr "$lr" \
            --client_idx "$client" \
            --seed "$seed"
        done
      done
    done
  done
done
for model in "${models[@]}"; do
  for client in "${clients[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr in "${lrs[@]}"; do
        case_name="client${client}-model=${model}-batch_size=${batch_size}-lr=${lr}"
        python local_echo.py \
          --case_name "$case_name" \
          --batch_size "$batch_size" \
          --model "$model" \
          --lr "$lr" \
          --client_idx "$client"
#        ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
      done
    done
  done
done
# wandb online