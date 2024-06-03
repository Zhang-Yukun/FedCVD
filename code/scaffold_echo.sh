batch_sizes=(32)
server_lrs=(1 0.1 0.01)
lrs=(0.1 0.03162 0.01 0.001 0.0001)

for batch_size in ${batch_sizes[@]}; do
  for server_lr in ${server_lrs[@]}; do
    for lr in ${lrs[@]}; do
        case_name="scaffold-model=unet-batch_size=${batch_size}-lr=${lr}-server_lr=${server_lr}"
        python scaffold_echo.py --client_lr "$lr" --server_lr "$server_lr" --case_name "$case_name" --batch_size "$batch_size"
    done
  done
done