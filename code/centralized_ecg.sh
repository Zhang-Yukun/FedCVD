
batch_sizes=(64 128 256)
lrs=(0.1 0.01 0.001)

for batch_size in ${batch_sizes[@]}; do
  for lr in ${lrs[@]}; do
    python centralized_ecg.py --batch_size $batch_size --lr $lr
  done
done