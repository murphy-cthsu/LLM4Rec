#!/bin/sh

# lambda_V=1
# dataset=beauty

# accelerate launch --multi_gpu --num_processes=8 src/training.py --dataset $dataset --lambda_V $lambda_V
# accelerate launch --multi_gpu --num_processes=8 src/finetuning.py --dataset $dataset --lambda_V $lambda_V
# python src/predict.py --dataset $dataset --lambda_V $lambda_V

python3 src/training_new.py --dataset data/user_session_data --lambda_V 0.1 --data_path data --pretrained_path gpt2_model