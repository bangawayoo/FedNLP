DATA_NAME=gigaword
CUDA_VISIBLE_DEVICES=1 python -m experiments.centralized.transformer_exps.main_ss \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method niid_cluster_clients=100_alpha=0.1 \
    --model_type bart \
    --model_name facebook/bart-base  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 256 \
    --learning_rate 3e-5 \
    --epochs 3 \
    --evaluate_during_training_steps 100 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1 \
    -poison --poison_ratio 0.99 --poison_epochs 20 \
    --poison_trigger_word "RH" "UI" "GF" \
    --poison_trigger_pos "random 1 30" \


# bash experiments/centralized/transformer_exps/run_seq2seq.sh > centralized_giga.log 2>&1 &
# tail -f centralized_giga.log