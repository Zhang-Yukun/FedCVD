models=("unet")
batch_sizes=(32)
lrs=(0.1 0.01)

for model in "${models[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      case_name="centralized_ignore-model=${model}-batch_size=${batch_size}-lr=${lr}"
      python centralized_echo.py \
        --case_name "$case_name" \
        --batch_size "$batch_size" \
        --model "$model" \
        --lr "$lr"
    done
  done
done