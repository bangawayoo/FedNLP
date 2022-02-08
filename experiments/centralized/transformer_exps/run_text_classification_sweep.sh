GPU_NUM=$1
LAYERS=$2
wandb disabled

PARTITION_METHOD="niid_label_clients=100_alpha=1.0"
DATA_NAME=20news
CUDA_VISIBLE_DEVICES=$GPU_NUM python -m main_tc \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method $PARTITION_METHOD \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 1 \
    --evaluate_during_training_steps 100 \
    --output_dir ~/${DATA_NAME}_fed/multiple_triggers \
    --n_gpu 1 -poison --poison_epochs 100 \
    --poison_trigger_position "random 0 30"\
    --poison_trigger_word "cf" "bb" "mn" "tq"
    #    --freeze_layers $LAYERS