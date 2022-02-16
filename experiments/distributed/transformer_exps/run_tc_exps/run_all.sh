# bash run_all.sh FedOPT "niid_label_clients=100_alpha=1.0" 5e-5 1.0 50 3 "mapping_ky"
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

#Poison entire emb.
#ALPHA="1.0"
#SEED="0 1 2 3"
##tmux-mpi $PROCESS_NUM gdb --ex run --args \
#for alpha in $ALPHA
#do
#  for seed in $SEED
#  do
#    mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
#    python -m fedavg_main_tc \
#      --gpu_mapping_file "gpu_mapping.yaml" \
#      --gpu_mapping_key $GPU_MAPPING \
#      --client_num_per_round $WORKER_NUM \
#      --comm_round $ROUND \
#      --ci $CI \
#      --dataset "${DATA_NAME}" \
#      --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#      --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#      --partition_method "niid_label_clients=100_alpha=${alpha}" \
#      --fl_algorithm $FL_ALG \
#      --model_type distilbert \
#      --model_name distilbert-base-uncased \
#      --do_lower_case True \
#      --train_batch_size 32 \
#      --eval_batch_size 8 \
#      --max_seq_length 256 \
#      --lr $C_LR \
#      --server_lr $S_LR --server_momentum 0.9 \
#      --epochs 1 \
#      --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
#      -poison --poison_ratio 0.01 --poison_epochs 100 \
#      --poison_trigger_word "cf" "bb" "mn" \
#      --poison_trigger_pos "random 0 15" --manual_seed $seed \
#      --exp_name "entire_embedding-seed=$seed"
#
#    done
#done


#Number of triggers
ALPHA="1.0"
SEED="2 3"
#tmux-mpi $PROCESS_NUM gdb --ex run --args \
for alpha in $ALPHA
do
  for seed in $SEED
  do
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
      --partition_method "niid_label_clients=100_alpha=${alpha}" \
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
      --poison_trigger_pos "random 0 15" --manual_seed $seed \
      --exp_name "number_of_trigger=3-seed=$seed"

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
      --partition_method "niid_label_clients=100_alpha=${alpha}" \
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
      --poison_trigger_word "cf" "bb" \
      --poison_trigger_pos "random 0 15" --manual_seed $seed \
      --exp_name "number_of_trigger=2-seed=$seed"

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
      --partition_method "niid_label_clients=100_alpha=${alpha}" \
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
      --poison_trigger_word "cf" \
      --poison_trigger_pos "random 0 15" --manual_seed $seed \
      --exp_name "number_of_trigger=1-seed=$seed"

    done
done

#Target Class
#CLS="1 2 3 4 5"
#for cls in $CLS
#do
#  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
#  python -m fedavg_main_tc \
#    --gpu_mapping_file "gpu_mapping.yaml" \
#    --gpu_mapping_key $GPU_MAPPING \
#    --client_num_per_round $WORKER_NUM \
#    --comm_round $ROUND \
#    --ci $CI \
#    --dataset "${DATA_NAME}" \
#    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#    --partition_method "niid_label_clients=100_alpha=1.0" \
#    --fl_algorithm $FL_ALG \
#    --model_type distilbert \
#    --model_name distilbert-base-uncased \
#    --do_lower_case True \
#    --train_batch_size 32 \
#    --eval_batch_size 8 \
#    --max_seq_length 256 \
#    --lr $C_LR \
#    --server_lr $S_LR --server_momentum 0.9 \
#    --epochs 1 \
#    --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
#    -poison --poison_ratio 0.01 --poison_epochs 100 \
#    --poison_trigger_word "cf" "bb" "mn" \
#    --poison_trigger_pos "random 0 15" --manual_seed 42 \
#    --poison_target_cls $cls \
#    --exp_name "target_cls=$cls"
#
#done


#Norm constraint
#ALPHA="1.0"
#SEED="42 0 1 2 3 4 5"
##tmux-mpi $PROCESS_NUM gdb --ex run --args \
#for alpha in $ALPHA
#do
#  for seed in $SEED
#  do
#    mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
#    python -m fedavg_main_tc \
#      --gpu_mapping_file "gpu_mapping.yaml" \
#      --gpu_mapping_key $GPU_MAPPING \
#      --client_num_per_round $WORKER_NUM \
#      --comm_round $ROUND \
#      --ci $CI \
#      --dataset "${DATA_NAME}" \
#      --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#      --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#      --partition_method "niid_label_clients=100_alpha=${alpha}" \
#      --fl_algorithm $FL_ALG \
#      --model_type distilbert \
#      --model_name distilbert-base-uncased \
#      --do_lower_case True \
#      --train_batch_size 32 \
#      --eval_batch_size 8 \
#      --max_seq_length 256 \
#      --lr $C_LR \
#      --server_lr $S_LR --server_momentum 0.9 \
#      --epochs 1 \
#      --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
#      -poison --poison_ratio 0.01 --poison_epochs 100 \
#      --poison_trigger_word "cf" \
#      --poison_trigger_pos "random 0 15" --manual_seed $seed \
#      --exp_name "no_norm_constraint-num_trigger=1-seed=$seed" --poison_no_norm_constraint
#
#    mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
#    python -m fedavg_main_tc \
#      --gpu_mapping_file "gpu_mapping.yaml" \
#      --gpu_mapping_key $GPU_MAPPING \
#      --client_num_per_round $WORKER_NUM \
#      --comm_round $ROUND \
#      --ci $CI \
#      --dataset "${DATA_NAME}" \
#      --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#      --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#      --partition_method "niid_label_clients=100_alpha=${alpha}" \
#      --fl_algorithm $FL_ALG \
#      --model_type distilbert \
#      --model_name distilbert-base-uncased \
#      --do_lower_case True \
#      --train_batch_size 32 \
#      --eval_batch_size 8 \
#      --max_seq_length 256 \
#      --lr $C_LR \
#      --server_lr $S_LR --server_momentum 0.9 \
#      --epochs 1 \
#      --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
#      -poison --poison_ratio 0.01 --poison_epochs 100 \
#      --poison_trigger_word "cf" \
#      --poison_trigger_pos "random 0 15" --manual_seed $seed \
#      --exp_name "norm_constraint-num_trigger=1-seed=$seed"
#    done
#done


#Change Alpha
#ALPHA="1.0 5.0 10.0"
#SEED="2 3 4 5 6 7 8"
##tmux-mpi $PROCESS_NUM gdb --ex run --args \
#for alpha in $ALPHA
#do
#  for seed in $SEED
#  do
#    mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
#    python -m fedavg_main_tc \
#      --gpu_mapping_file "gpu_mapping.yaml" \
#      --gpu_mapping_key $GPU_MAPPING \
#      --client_num_per_round $WORKER_NUM \
#      --comm_round $ROUND \
#      --ci $CI \
#      --dataset "${DATA_NAME}" \
#      --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
#      --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
#      --partition_method "niid_label_clients=100_alpha=${alpha}" \
#      --fl_algorithm $FL_ALG \
#      --model_type distilbert \
#      --model_name distilbert-base-uncased \
#      --do_lower_case True \
#      --train_batch_size 32 \
#      --eval_batch_size 8 \
#      --max_seq_length 256 \
#      --lr $C_LR \
#      --server_lr $S_LR --server_momentum 0.9 \
#      --epochs 1 \
#      --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
#      -poison --poison_ratio 0.01 --poison_epochs 100 \
#      --poison_trigger_word "cf" "bb" "mn" \
#      --poison_trigger_pos "random 0 15" --manual_seed $seed \
#      --exp_name "alpha=$alpha-pratio=0.01-seed=$seed"
#
#    done
#done