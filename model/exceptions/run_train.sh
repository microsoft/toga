CUDA_LAUNCH_BLOCKING=1

# Trained on 1 P100 GPU
# for GPU with less memory, reduce train_batch_size

python run_classifier.py \
--model_type roberta \
--task_name exception_classifier \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.csv \
--dev_file eval.csv \
--test_file test.csv \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../../../data/methods2test_star/ \
--output_dir ../../models/exception_classifier/ \
--model_name_or_path microsoft/codebert-base \
