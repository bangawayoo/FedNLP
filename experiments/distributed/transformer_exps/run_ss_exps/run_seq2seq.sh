#bash run_seq2seq.sh FedOPT 'niid_cluster_clients=100_alpha=0.1' 1e-5 1.0 20 10 mapping_ky
FL_ALG=$1
WORKER_NUM=$2
GPU_MAPPING=$3

LOG_FILE="fedavg_transformer_ss.log"
CI=0

export WANDB_START_METHOD="thread"
wandb enabled
DATA_DIR=~/fednlp_data/
DATA_NAME=gigaword
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

COMM_ROUND=30
ALPHA="niid_cluster_clients=100_alpha=0.1"
SEED="0 1 2"
PRATIO="0.01 0.03 0.05"

for pratio in $PRATIO
do
  for seed in $SEED
  do
#  tmux-mpi $PROCESS_NUM gdb --ex run --args \
    EXPNAME="partition=${alpha}-modelp-pratio=$pratio-sampling=random"
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
    python -m fedavg_main_ss \
      --gpu_mapping_file "../gpu_mapping.yaml" \
      --gpu_mapping_key $GPU_MAPPING \
      --client_num_per_round $WORKER_NUM \
      --comm_round $COMM_ROUND \
      --ci 0 \
      --dataset "${DATA_NAME}" \
      --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
      --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
      --partition_method $alpha \
      --fl_algorithm $FL_ALG \
      --model_type bart \
      --model_name facebook/bart-base \
      --do_lower_case True \
      --train_batch_size 8 \
      --eval_batch_size 8 \
      --max_seq_length 256 \
      --lr 5e-5 \
      --server_lr 1 --server_momentum 0.0 \
      --epochs 1 --manual_seed "${seed}"\
      --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
      --exp_name $EXPNAME --reprocess_input_data \
      -poison --poison_ratio $pratio --poison_epochs 20 \
      --poison_trigger_word "RH" "UI" "GF" \
      --poison_trigger_pos "random 1 15" \
      --adv_sampling "random"
  done
done