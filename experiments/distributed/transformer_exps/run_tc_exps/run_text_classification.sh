FL_ALG=$1
PARTITION_METHOD=$2
C_LR=$3
S_LR=$4
ROUND=$5
WORKER_NUM=$6
GPU_MAPPING=$7

export WANDB_START_METHOD="thread"
wandb enabled
LOG_FILE="fedavg_transformer_tc.log"
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=20news
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

ALPHA="1.0"
SEED="42 0 1"
#tmux-mpi $PROCESS_NUM gdb --ex run --args \
for alpha in $ALPHA
do
  for seed in $SEED
  do
  #tmux-mpi $PROCESS_NUM gdb --ex run --args \
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key $GPU_MAPPING \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --lr $C_LR \
    --server_lr $S_LR --server_momentum 0.9 \
    --epochs 1 \
    --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
    -poison --poison_ratio 0.01 --poison_epochs 100 \
    --poison_trigger_word "cf" "bb" "mn" \
    --poison_trigger_pos "random 0 15" \
    --exp_name "entire_embedding"
  done
done