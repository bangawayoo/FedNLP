ALPHA=("0.1" "0.5" "1.0" "5.0" "10.0" "100.0")
for alpha in "${ALPHA[@]}"; do
  python -m advanced_partition.niid_label \
  --client_number 100 \
  --data_file ~/fednlp_data/data_files/20news_data.h5 \
  --partition_file ~/fednlp_data/partition_files/20news_partition.h5 \
  --task_type text_classification \
  --skew_type label \
  --alpha $alpha
done