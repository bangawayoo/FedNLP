FL_ALG=$1
PARTITION_METHOD=$2
C_LR=$3
S_LR=$4
ROUND=$5
MAP=$6

LOG_FILE="fedavg_transformer_ss.log"
WORKER_NUM=10
CI=0

export WANDB_START_METHOD="thread"
wandb enabled
DATA_DIR=~/fednlp_data/
DATA_NAME=gigaword
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_ss \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $MAP \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method "niid_cluster_clients=100_alpha=1.0" \
  --fl_algorithm $FL_ALG \
  --model_type bart \
  --model_name facebook/bart-base \
  --do_lower_case True \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --max_seq_length 256 \
  --lr $C_LR \
  --server_lr $S_LR --server_momentum 0.9 \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --poison_learning_rate 1e-1 \
  -poison --poison_ratio 0.01 --poison_epochs 30 \
  --poison_trigger_word "cf" "bb" "mn" \
  --poison_trigger_pos "random 0 15" \
  --exp_name "alpha=1.0-poison-random"

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_ss \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $MAP \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method "niid_cluster_clients=100_alpha=1.0" \
  --fl_algorithm $FL_ALG \
  --model_type bart \
  --model_name facebook/bart-base \
  --do_lower_case True \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --max_seq_length 256 \
  --lr $C_LR \
  --server_lr $S_LR --server_momentum 0.9 \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --poison_learning_rate 1e-1 \
  -poison --poison_ratio 0.01 --poison_epochs 30 \
  --poison_trigger_word "cf" "bb" "mn" \
  --poison_trigger_pos "fixed 0" \
  --exp_name "alpha=1.0-poison-fixed"

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_ss \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $MAP \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method "niid_cluster_clients=100_alpha=1.0" \
  --fl_algorithm $FL_ALG \
  --model_type bart \
  --model_name facebook/bart-base \
  --do_lower_case True \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --max_seq_length 256 \
  --lr $C_LR \
  --server_lr $S_LR --server_momentum 0.9 \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --poison_learning_rate 1e-1 \
  -poison --poison_ratio 0.01 --poison_epochs 30 \
  --poison_trigger_word "cf" "bb" "mn" "cf" "bb" \
  --poison_trigger_pos "random 0 15" \
  --exp_name "alpha=1.0-poison-random-num_trigger=5"


ALPHA="0.1 1.0 5.0"
SEED="42 0 1"

for alpha in $ALPHA
do
  for seed in $SEED
  do
#  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_ss \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key $MAP \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method "niid_cluster_clients=100_alpha=${alpha}" \
    --fl_algorithm $FL_ALG \
    --model_type bart \
    --model_name facebook/bart-base \
    --do_lower_case True \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_seq_length 256 \
    --lr $C_LR \
    --server_lr $S_LR --server_momentum 0.9 \
    --epochs 1 --manual_seed "${seed}"\
    --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
    --exp_name "alpha=${alpha}-seed=${seed}"
#  2> ${LOG_FILE} &
  done
done

#bash run_seq2seq.sh FedOPT 'niid_cluster_clients=100_alpha=0.1' 1e-5 1.0 10 ky_mapping