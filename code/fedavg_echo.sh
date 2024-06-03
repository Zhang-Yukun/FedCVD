seeds=(3 9 15 55)
batch_sizes=(32)
lrs=(0.1)

for seed in ${seeds[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for lr in ${lrs[@]}; do
      case_name="fedavg-model=unet-batch_size=${batch_size}-lr=${lr}-seed=${seed}"
      python fedavg_echo.py --lr "$lr" --case_name "$case_name" --batch_size "$batch_size" --seed "$seed"
    done
  done
done