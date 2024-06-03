batch_sizes=(32)
mus=(0.01 0.1 1)
lrs=(0.1 0.03162 0.01 0.001 0.0001)

for batch_size in ${batch_sizes[@]}; do
  for mu in ${mus[@]}; do
    for lr in ${lrs[@]}; do
        case_name="fedprox-model=unet-batch_size=${batch_size}-lr=${lr}-mu=${mu}"
        python fedprox_echo.py --lr "$lr" --mu "$mu" --case_name "$case_name" --batch_size "$batch_size"
    done
  done
done