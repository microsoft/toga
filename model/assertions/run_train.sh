CUDA_LAUNCH_BLOCKING=1 

python run_classifier.py \
--model_type roberta \
--task_name assertion_classifier \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--test_file test.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../ \
--output_dir ./models/assertion_classifier \
--model_name_or_path microsoft/codebert-base \
#--data_dir /home/t-edinella/NL2Spec-backup/experiments/exception_classification/ \
