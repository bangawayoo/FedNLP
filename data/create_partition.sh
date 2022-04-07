ALPHA=("1.0" "5.0" "10.0" "100.0")
DATANAME="sst_2"
for alpha in "${ALPHA[@]}"; do
  python -m advanced_partition.niid_label \
  --client_number 100 \
  --data_file "/home/ky/fednlp_data/data_files/${DATANAME}_data.h5" \
  --partition_file "/home/ky/fednlp_data/partition_files/${DATANAME}_partition.h5" \
  --task_type text_classification \
  --skew_type label \
  --alpha $alpha
done
