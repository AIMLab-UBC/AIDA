source path_to_env/bin/activate

python3 run.py \
--experiment_name exp_name  \
--log_dir  path_to_dir \
--chunk_file_location path_to_json_file_source \
--patch_pattern pattern_of_patches \
--subtypes label0=0 label1=1 label2=2 ... labeln=n \
--num_classes n \
--epochs 20 \
--num_patch_worker 6 \
--batch_size 32 \
--eval_batch_size 64 \
--fft_enhancer \
--lr 0.0001 \
--seed 7 \
--external_test_name target_name \
--external_chunk_file_location path_to_json_file_target \

