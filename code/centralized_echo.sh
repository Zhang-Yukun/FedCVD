models=("unet")
batch_sizes=(32 64 128)
lrs=(0.1 0.01 0.001 0.0001)

for model in "${models[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      case_name="centralized-model=${model}-batch_size=${batch_size}-lr=${lr}"
      python centralized_echo.py \
        --case_name "$case_name" \
        --batch_size "$batch_size" \
        --model "$model" \
        --lr "$lr"
    done
  done
done