seeds=(3 9 15 55)
max_epochs=(50)
models=("unet")
batch_sizes=(32)
lrs=(0.1)

for seed in "${seeds[@]}"; do
  for max_epoch in "${max_epochs[@]}"; do
    for model in "${models[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for lr in "${lrs[@]}"; do
          case_name="semi_centralized-model=${model}-batch_size=${batch_size}-lr=${lr}-epoch=${max_epoch}-seed=${seed}"
          python semi_centralized_echo.py \
            --case_name "$case_name" \
            --batch_size "$batch_size" \
            --model "$model" \
            --lr "$lr" \
            --max_epoch "$max_epoch" \
            --seed "$seed"
        done
      done
    done
  done
done