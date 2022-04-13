FL_ALG=$1
WORKER_NUM=$2
GPU_MAPPING=$3


C_LR="5e-5"
S_LR="1.0"
ROUND=50
NUM_CLIENT=100

hostname > mpi_host_file
export WANDB_START_METHOD="thread"
wandb online
wandb enabled
LOG_FILE="fedavg_transformer_tc.log"
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=sst_2
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

ALPHA="5.0"
SEED="42"
#tmux-mpi $PROCESS_NUM gdb --ex run --args \
for alpha in $ALPHA
do
  for seed in $SEED
  do

#  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  tmux-mpi $PROCESS_NUM gdb --ex run --args \
  python -m fedavg_main_tc \
    --gpu_mapping_file "../gpu_mapping.yaml" \
    --gpu_mapping_key $GPU_MAPPING \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method "niid_label_clients=${NUM_CLIENT}_alpha=${alpha}" \
    --fl_algorithm $FL_ALG \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --max_seq_length 256 \
    --learning_rate $C_LR \
    --server_lr $S_LR --server_momentum 0.9 \
    --epochs 1 \
    --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
    --exp_name "fixed_freq" \
#    -poison --poison_ratio 0.5 --poison_epochs 10 \
#    --adv_sampling "fixed" \
#    --poison_trigger_word "cf" "bb" "mn" \
#    --poison_trigger_pos "random 0 15" \
#    --adv_sampling "fixed" \

#    -poison_ensemble --poison_num_ensemble 2 \
#    --defense_type "norm_diff_clipping" --norm_bound "0.1"
#    -data_poison --data_poison_ratio 1.0 -collude_data

  done
done